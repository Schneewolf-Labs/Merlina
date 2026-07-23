#!/usr/bin/env python3
"""
Tests for the unified-memory safety guard (src/memory_guard.py).

Covers detection heuristics, allocator-cap math, the watchdog's
soft/hard escalation state machine, forensic logging, thread abort,
and the TrainingMemoryGuard facade used by the runners.

Runs standalone (python tests/test_memory_guard.py) or under pytest.
No GPU required — torch is faked where needed.
"""

import os
import sys
import threading
import time
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock  # noqa: E402 — needed before src import below

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Standalone-run support: importing src triggers torch via preflight_checks.
# Under pytest, conftest.py installs these mocks; mirror the minimal set here
# so `python tests/test_memory_guard.py` also works without ML deps.
if 'torch' not in sys.modules:
    try:
        import torch  # noqa: F401
    except ImportError:
        _mock_torch = mock.MagicMock()
        _mock_torch.__spec__ = mock.MagicMock()
        _mock_torch.cuda.is_available.return_value = False
        sys.modules['torch'] = _mock_torch

from src.memory_guard import (
    MemoryPressureAbort,
    MemorySample,
    MemoryWatchdog,
    TrainingMemoryGuard,
    UnifiedMemoryInfo,
    _abort_thread,
    compute_allocator_fraction,
    detect_unified_memory,
    effective_reserve_gb,
)


def _sample(available_gb: float, total_gb: float = 128.0) -> MemorySample:
    return MemorySample(
        timestamp=time.time(),
        system_available=available_gb,
        system_used=total_gb - available_gb,
        system_total=total_gb,
    )


def _fake_torch(device_name: str, gpu_total_gb: float, cuda_available: bool = True):
    """A minimal torch stand-in for detection tests."""
    props = SimpleNamespace(name=device_name, total_memory=int(gpu_total_gb * 1024**3))
    cuda = SimpleNamespace(
        is_available=lambda: cuda_available,
        get_device_properties=lambda device: props,
    )
    return SimpleNamespace(cuda=cuda)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_detect_override_forces_result():
    info = detect_unified_memory(override="true")
    assert info.is_unified is True
    assert "forced" in info.reason

    info = detect_unified_memory(override="false")
    assert info.is_unified is False


def test_detect_by_gpu_name():
    with mock.patch.dict(sys.modules, {"torch": _fake_torch("NVIDIA GB10", 119.0)}):
        info = detect_unified_memory(override="auto")
    assert info.is_unified is True
    assert "gb10" in info.reason.lower()

    with mock.patch.dict(sys.modules, {"torch": _fake_torch("NVIDIA GH200 480GB", 96.0)}):
        info = detect_unified_memory(override="auto")
    assert info.is_unified is True


def test_detect_by_shared_pool_ratio():
    # GPU claims ~all of system RAM -> unified, regardless of name
    fake = _fake_torch("Unknown SoC iGPU", 120.0)
    vm = SimpleNamespace(total=128 * 1024**3, available=100 * 1024**3, used=28 * 1024**3)
    with mock.patch.dict(sys.modules, {"torch": fake}), \
         mock.patch("src.memory_guard.psutil.virtual_memory", return_value=vm):
        info = detect_unified_memory(override="auto")
    assert info.is_unified is True
    assert "shared pool" in info.reason


def test_detect_discrete_gpu_is_not_unified():
    # 24GB card in a 128GB box -> discrete
    fake = _fake_torch("NVIDIA GeForce RTX 4090", 24.0)
    vm = SimpleNamespace(total=128 * 1024**3, available=100 * 1024**3, used=28 * 1024**3)
    with mock.patch.dict(sys.modules, {"torch": fake}), \
         mock.patch("src.memory_guard.psutil.virtual_memory", return_value=vm):
        info = detect_unified_memory(override="auto")
    assert info.is_unified is False


def test_detect_no_cuda():
    fake = _fake_torch("n/a", 0.0, cuda_available=False)
    with mock.patch.dict(sys.modules, {"torch": fake}):
        info = detect_unified_memory(override="auto")
    assert info.is_unified is False


# ---------------------------------------------------------------------------
# Allocator-cap math
# ---------------------------------------------------------------------------

def test_effective_reserve_caps_at_quarter_of_pool():
    # 12GB reserve on a 128GB Spark: applied as-is
    assert effective_reserve_gb(128.0, 12.0) == 12.0
    # 12GB reserve on a 16GB Jetson: capped to 25% of pool
    assert effective_reserve_gb(16.0, 12.0) == 4.0


def test_compute_allocator_fraction():
    # 128GB pool, 12GB reserve -> ~90.6%
    frac = compute_allocator_fraction(128.0, 12.0)
    assert abs(frac - (116.0 / 128.0)) < 1e-9
    # Never below the 0.3 floor, never above 1.0
    assert compute_allocator_fraction(1.0, 100.0) >= 0.3
    assert compute_allocator_fraction(100.0, 0.0) == 1.0


# ---------------------------------------------------------------------------
# Watchdog state machine
# ---------------------------------------------------------------------------

def _make_watchdog(readings, log_path=None, soft=8.0, hard=3.0):
    """Watchdog fed from a scripted list of free-GB readings."""
    it = iter(readings)
    state = {"current": readings[0]}

    def reader():
        try:
            state["current"] = next(it)
        except StopIteration:
            pass
        return _sample(state["current"])

    soft_hits, hard_hits = [], []
    wd = MemoryWatchdog(
        soft_free_gb=soft,
        hard_free_gb=hard,
        poll_seconds=999,  # never self-polls; tests drive check_once()
        on_soft_limit=soft_hits.append,
        on_hard_limit=hard_hits.append,
        log_path=log_path,
        memory_reader=reader,
        job_id="job_test",
    )
    return wd, soft_hits, hard_hits


def test_watchdog_soft_then_hard_fire_once_each():
    wd, soft_hits, hard_hits = _make_watchdog([50, 50, 7.5, 7.0, 2.5, 2.0, 1.0])
    wd.check_once()  # 50 (consumed at init) -> actually first check reads 50
    wd.check_once()  # 7.5 -> soft
    wd.check_once()  # 7.0 -> already triggered, no repeat
    assert len(soft_hits) == 1 and len(hard_hits) == 0
    wd.check_once()  # 2.5 -> hard
    wd.check_once()  # 2.0 -> no repeat
    wd.check_once()
    assert len(soft_hits) == 1
    assert len(hard_hits) == 1
    assert wd.soft_triggered and wd.hard_triggered


def test_watchdog_hard_directly_without_soft():
    # A fast spike can blow straight through the soft floor between polls
    wd, soft_hits, hard_hits = _make_watchdog([50, 1.0])
    wd.check_once()
    wd.check_once()
    assert len(hard_hits) == 1
    assert len(soft_hits) == 0  # soft handler skipped, hard escalation owns it
    assert wd.soft_triggered  # but marked so a later soft doesn't re-fire


def test_watchdog_scales_floors_on_small_systems():
    def reader():
        return _sample(10.0, total_gb=16.0)

    wd = MemoryWatchdog(soft_free_gb=8.0, hard_free_gb=3.0, memory_reader=reader)
    # 8GB soft floor on a 16GB board would trip immediately; scaled to 10%/4%
    assert wd.soft_free_gb == 16.0 * 0.10
    assert wd.hard_free_gb == 16.0 * 0.04


def test_watchdog_writes_forensic_log():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "memory_guard.log"
        wd, _, _ = _make_watchdog([50, 20], log_path=log_path)
        wd.check_once()
        wd.check_once()
        content = log_path.read_text()
        lines = [l for l in content.splitlines() if l.strip()]
        assert len(lines) == 2
        assert "[job_test]" in lines[0]
        assert "sys_avail=" in lines[0] and "cuda_alloc=" in lines[0]


# ---------------------------------------------------------------------------
# Thread abort
# ---------------------------------------------------------------------------

def test_abort_thread_raises_in_target():
    caught = {}

    def victim():
        try:
            while True:
                time.sleep(0.01)
        except MemoryPressureAbort as e:
            caught["exc"] = e

    t = threading.Thread(target=victim, daemon=True)
    t.start()
    time.sleep(0.05)
    assert _abort_thread(t, MemoryPressureAbort) is True
    t.join(timeout=5)
    assert not t.is_alive()
    assert isinstance(caught["exc"], MemoryPressureAbort)
    # Default message explains the unified-memory situation
    assert "unified-memory" in str(caught["exc"])


def test_abort_dead_thread_returns_false():
    t = threading.Thread(target=lambda: None)
    t.start()
    t.join()
    assert _abort_thread(t, MemoryPressureAbort) is False


# ---------------------------------------------------------------------------
# TrainingMemoryGuard facade
# ---------------------------------------------------------------------------

def _guard_settings(**overrides):
    defaults = dict(
        memory_guard_enabled=True,
        unified_memory="true",
        memory_guard_reserve_gb=12.0,
        memory_guard_soft_free_gb=8.0,
        memory_guard_hard_free_gb=3.0,
        memory_guard_poll_seconds=999.0,
        memory_guard_log_enabled=False,
        data_dir=Path("./data"),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_guard_inert_on_discrete_systems():
    guard = TrainingMemoryGuard("job_x", settings=_guard_settings(unified_memory="false"))
    guard.install()
    assert guard.active is False
    # All lifecycle calls must be safe no-ops
    guard.set_trainer(object())
    guard.shutdown()


def test_guard_inert_when_disabled():
    guard = TrainingMemoryGuard(
        "job_x", settings=_guard_settings(memory_guard_enabled=False)
    )
    guard.install()
    assert guard.active is False
    guard.shutdown()


def test_guard_active_and_soft_stop_requests_trainer_stop():
    events = []
    guard = TrainingMemoryGuard(
        "job_x",
        notify=lambda level, msg: events.append((level, msg)),
        settings=_guard_settings(),
        memory_reader=lambda: _sample(50.0),
    )
    guard.install()
    try:
        assert guard.active is True
        assert guard._watchdog is not None and guard._watchdog.is_alive()

        trainer = mock.MagicMock()
        guard.set_trainer(trainer)
        guard._handle_soft_limit(_sample(5.0))
        trainer.request_stop.assert_called_once()
        assert events and events[0][0] == "warning"
        assert guard.pressure_event is not None
    finally:
        guard.shutdown()
    assert guard.active is False


def test_guard_soft_before_trainer_does_not_crash():
    guard = TrainingMemoryGuard(
        "job_x", settings=_guard_settings(), memory_reader=lambda: _sample(50.0)
    )
    guard.install()
    try:
        guard._handle_soft_limit(_sample(5.0))  # no trainer yet: warn only
        assert guard.pressure_event is not None
    finally:
        guard.shutdown()


def test_guard_hard_limit_aborts_training_thread():
    caught = {}
    installed = threading.Event()
    release = threading.Event()

    def fake_training_thread():
        guard = TrainingMemoryGuard(
            "job_x", settings=_guard_settings(), memory_reader=lambda: _sample(50.0)
        )
        guard.install()  # captures this thread as the abort target
        caught["guard"] = guard
        installed.set()
        try:
            while not release.is_set():
                time.sleep(0.01)
            caught["result"] = "completed"
        except MemoryPressureAbort as e:
            caught["result"] = e
        finally:
            guard.shutdown()

    t = threading.Thread(target=fake_training_thread, daemon=True)
    t.start()
    assert installed.wait(timeout=5)
    caught["guard"]._handle_hard_limit(_sample(1.0))
    t.join(timeout=5)
    assert isinstance(caught["result"], MemoryPressureAbort)


if __name__ == "__main__":
    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"✅ {name}")
        except Exception as e:
            failed += 1
            print(f"❌ {name}: {e}")
    print("=" * 60)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
