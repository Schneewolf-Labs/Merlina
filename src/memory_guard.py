"""
Unified-memory safety guard (DGX Spark / Grace-Blackwell, GH200, Jetson).

On a discrete GPU, a training memory spike raises a clean, catchable CUDA
OOM. On unified-memory machines (NVIDIA GB10 "DGX Spark", GB200/GH200
Grace superchips, Jetson Orin/Thor) the GPU and system RAM share one
physical pool: the CUDA allocator will keep growing into memory the OS
needs, and instead of an OOM the whole box locks up or the kernel
OOM-killer shoots arbitrary processes. A 12B QLoRA that would fail
gracefully on a 4090 can take the entire machine down on a Spark.

Three defenses, all wired through this module:

1. **Allocator cap** — ``apply_memory_limits()`` calls
   ``torch.cuda.set_per_process_memory_fraction()`` so PyTorch raises an
   ordinary CUDA OOM *before* the shared pool is exhausted. The job fails
   cleanly; the machine survives.

2. **Watchdog thread** — ``MemoryWatchdog`` samples system + CUDA memory.
   Below a soft free-RAM floor it requests a graceful trainer stop (the
   checkpoint is saved); below a hard floor it aborts the training thread
   with :class:`MemoryPressureAbort` (host-side spikes — full-precision
   loads, dataset mapping, leaks — never go through the CUDA allocator,
   so the cap alone can't catch them).

3. **Forensic log** — every sample is appended and fsync'd to
   ``data/memory_guard.log`` so if the box does go down there is a record
   of the memory picture right before death.

Everything is inert on machines that are not unified-memory (override via
``UNIFIED_MEMORY=true/false`` in ``.env``). Runners use the high-level
:class:`TrainingMemoryGuard` facade; detection helpers are also used by
pre-flight checks.
"""

import ctypes
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import psutil

logger = logging.getLogger(__name__)

# GPU name fragments that identify unified-memory (SoC / superchip) parts.
# Matched case-insensitively against torch's device name.
UNIFIED_MEMORY_GPU_PATTERNS = (
    "gb10",      # DGX Spark
    "gb200",     # Grace-Blackwell superchip
    "gh200",     # Grace-Hopper superchip
    "grace",
    "jetson",
    "orin",
    "thor",
)

# If the GPU reports at least this fraction of total system RAM as its own
# memory, GPU and system are one pool (a discrete card never comes close).
UNIFIED_POOL_RATIO = 0.9

# The allocator cap never reserves more than this fraction of the pool for
# the host, so small unified boards (16 GB Jetson) still get to train.
MAX_RESERVE_FRACTION = 0.25

# Fallback defaults, overridable via Settings (config.py) / .env.
DEFAULT_RESERVE_GB = 12.0
DEFAULT_SOFT_FREE_GB = 8.0
DEFAULT_HARD_FREE_GB = 3.0
DEFAULT_POLL_SECONDS = 5.0

_FORENSIC_LOG_MAX_BYTES = 2 * 1024 * 1024


class MemoryPressureAbort(RuntimeError):
    """Raised inside the training thread when system memory pressure is
    critical and a graceful stop is no longer an option.

    Injected asynchronously via ``PyThreadState_SetAsyncExc``, which only
    accepts an exception *class* — so the explanatory message lives in the
    no-args constructor.
    """

    def __init__(self, *args):
        if not args:
            args = (
                "Aborted by Merlina's memory guard: free system memory fell below "
                "the critical floor on a unified-memory machine (GPU and system RAM "
                "share one pool — continuing would crash the whole machine, not just "
                "this job). See data/memory_guard.log for the memory timeline. "
                "Reduce max_length, batch_size, enable 4-bit quantization / "
                "gradient checkpointing, or lower MEMORY_GUARD_RESERVE_GB.",
            )
        super().__init__(*args)


@dataclass
class UnifiedMemoryInfo:
    """Result of unified-memory detection."""
    is_unified: bool
    reason: str
    device_name: Optional[str] = None
    gpu_total_gb: float = 0.0
    system_total_gb: float = 0.0

    def to_dict(self) -> dict:
        return {
            "is_unified": self.is_unified,
            "reason": self.reason,
            "device_name": self.device_name,
            "gpu_total_gb": round(self.gpu_total_gb, 1),
            "system_total_gb": round(self.system_total_gb, 1),
        }


def _settings_override() -> Optional[str]:
    """Read the unified_memory override from Settings without requiring it."""
    try:
        from config import get_settings
        return getattr(get_settings(), "unified_memory", None)
    except Exception:
        return os.getenv("UNIFIED_MEMORY")


def detect_unified_memory(
    override: Optional[str] = None,
    device: int = 0,
) -> UnifiedMemoryInfo:
    """
    Detect whether GPU and system RAM share one physical pool.

    Signals, in order:
    1. Explicit override (``UNIFIED_MEMORY=true/false`` in .env, or the
       ``override`` argument) — always wins.
    2. GPU name matches a known unified-memory part (GB10, GH200, Jetson…).
    3. The GPU claims >= 90% of total system RAM as device memory —
       definitive for a shared pool, impossible for a discrete card.
    """
    system_total_gb = psutil.virtual_memory().total / (1024 ** 3)

    if override is None:
        override = _settings_override()
    if override is not None and str(override).strip().lower() not in ("", "auto"):
        forced = str(override).strip().lower() in ("1", "true", "yes", "on")
        return UnifiedMemoryInfo(
            is_unified=forced,
            reason=f"forced via UNIFIED_MEMORY={override}",
            system_total_gb=system_total_gb,
        )

    try:
        import torch
        if not torch.cuda.is_available():
            return UnifiedMemoryInfo(
                is_unified=False, reason="no CUDA device",
                system_total_gb=system_total_gb,
            )
        props = torch.cuda.get_device_properties(device)
        device_name = props.name
        gpu_total_gb = props.total_memory / (1024 ** 3)
    except Exception as e:  # torch missing / driver hiccup — assume discrete
        return UnifiedMemoryInfo(
            is_unified=False, reason=f"detection unavailable: {e}",
            system_total_gb=system_total_gb,
        )

    name_lower = device_name.lower()
    for pattern in UNIFIED_MEMORY_GPU_PATTERNS:
        if pattern in name_lower:
            return UnifiedMemoryInfo(
                is_unified=True,
                reason=f"GPU name '{device_name}' matches unified-memory part '{pattern}'",
                device_name=device_name,
                gpu_total_gb=gpu_total_gb,
                system_total_gb=system_total_gb,
            )

    if system_total_gb > 0 and gpu_total_gb >= UNIFIED_POOL_RATIO * system_total_gb:
        return UnifiedMemoryInfo(
            is_unified=True,
            reason=(
                f"GPU reports {gpu_total_gb:.0f}GB of {system_total_gb:.0f}GB "
                "system RAM as device memory (shared pool)"
            ),
            device_name=device_name,
            gpu_total_gb=gpu_total_gb,
            system_total_gb=system_total_gb,
        )

    return UnifiedMemoryInfo(
        is_unified=False,
        reason=f"discrete GPU ({device_name}, {gpu_total_gb:.0f}GB)",
        device_name=device_name,
        gpu_total_gb=gpu_total_gb,
        system_total_gb=system_total_gb,
    )


def effective_reserve_gb(total_pool_gb: float, configured_reserve_gb: float) -> float:
    """Host-side reserve actually applied: the configured value, but never
    more than 25% of the pool (a 12 GB reserve is right for a 128 GB Spark
    and absurd for a 16 GB Jetson)."""
    if total_pool_gb <= 0:
        return configured_reserve_gb
    return min(configured_reserve_gb, total_pool_gb * MAX_RESERVE_FRACTION)


def compute_allocator_fraction(total_pool_gb: float, reserve_gb: float) -> float:
    """CUDA allocator cap as a fraction of device-visible memory."""
    if total_pool_gb <= 0:
        return 1.0
    reserve = effective_reserve_gb(total_pool_gb, reserve_gb)
    return max(0.3, min(1.0, (total_pool_gb - reserve) / total_pool_gb))


def apply_memory_limits(
    info: Optional[UnifiedMemoryInfo] = None,
    reserve_gb: Optional[float] = None,
) -> Optional[float]:
    """
    On a unified-memory system, cap the CUDA caching allocator so a training
    spike raises a clean CUDA OOM instead of starving the OS.

    Returns the applied fraction, or None when nothing was applied
    (non-unified system, no CUDA, or guard disabled).
    """
    if info is None:
        info = detect_unified_memory()
    if not info.is_unified:
        return None

    if reserve_gb is None:
        try:
            from config import get_settings
            reserve_gb = float(get_settings().memory_guard_reserve_gb)
        except Exception:
            reserve_gb = DEFAULT_RESERVE_GB

    try:
        import torch
        if not torch.cuda.is_available():
            return None

        # Expandable segments cut fragmentation, which on a shared pool
        # directly translates to host RAM given back to the OS. Best-effort:
        # the private setter exists on torch >= 2.1; ignore if it moved.
        try:
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
            logger.info("Memory guard: enabled expandable_segments for the CUDA allocator")
        except Exception as e:
            logger.debug(f"Could not enable expandable_segments: {e}")

        fraction = None
        for device in range(torch.cuda.device_count()):
            total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            fraction = compute_allocator_fraction(total_gb, reserve_gb)
            torch.cuda.set_per_process_memory_fraction(fraction, device)
            logger.info(
                f"Memory guard: capped CUDA allocator on device {device} at "
                f"{fraction:.0%} of {total_gb:.0f}GB "
                f"(~{effective_reserve_gb(total_gb, reserve_gb):.0f}GB reserved for the OS) — "
                f"{info.reason}"
            )
        return fraction
    except Exception as e:
        logger.warning(f"Memory guard: could not apply allocator cap: {e}")
        return None


def _abort_thread(thread: threading.Thread, exc_type: type) -> bool:
    """Asynchronously raise ``exc_type`` in ``thread`` (CPython only).

    Best-effort: the exception surfaces at the next bytecode boundary, so a
    thread stuck deep inside a C extension may not see it immediately — but
    for the host-RAM spikes this guards against (Python-driven load loops,
    dataset mapping) it lands fast enough to save the machine.
    """
    if thread is None or not thread.is_alive():
        return False
    tid = thread.ident
    if tid is None:
        return False
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid), ctypes.py_object(exc_type)
    )
    if res > 1:
        # Undo: we hit more than one thread state, which must never happen.
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)
        return False
    return res == 1


@dataclass
class MemorySample:
    """One watchdog observation, in GB."""
    timestamp: float
    system_available: float
    system_used: float
    system_total: float
    cuda_allocated: float = 0.0
    cuda_reserved: float = 0.0

    def format_line(self) -> str:
        return (
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))} "
            f"sys_avail={self.system_available:.2f}GB "
            f"sys_used={self.system_used:.2f}GB/{self.system_total:.0f}GB "
            f"cuda_alloc={self.cuda_allocated:.2f}GB "
            f"cuda_reserved={self.cuda_reserved:.2f}GB"
        )


def _default_memory_reader() -> MemorySample:
    vm = psutil.virtual_memory()
    sample = MemorySample(
        timestamp=time.time(),
        system_available=vm.available / (1024 ** 3),
        system_used=vm.used / (1024 ** 3),
        system_total=vm.total / (1024 ** 3),
    )
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            sample.cuda_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            sample.cuda_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    except Exception:
        pass
    return sample


class MemoryWatchdog(threading.Thread):
    """
    Background sampler with two escalation levels.

    * ``system_available < soft_free_gb`` → ``on_soft_limit()`` once.
      Wired to a graceful ``trainer.request_stop()`` so the checkpoint is
      saved and the job ends as "stopped".
    * ``system_available < hard_free_gb`` → ``on_hard_limit()`` once.
      Wired to aborting the training thread — losing the job is the
      accepted cost of keeping the machine alive.

    Every sample is appended (and fsync'd) to ``log_path`` so a post-reboot
    investigation has the timeline a crashed kernel can't give you.
    """

    def __init__(
        self,
        soft_free_gb: float = DEFAULT_SOFT_FREE_GB,
        hard_free_gb: float = DEFAULT_HARD_FREE_GB,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        on_soft_limit: Optional[Callable[[MemorySample], None]] = None,
        on_hard_limit: Optional[Callable[[MemorySample], None]] = None,
        log_path: Optional[Path] = None,
        memory_reader: Callable[[], MemorySample] = _default_memory_reader,
        job_id: Optional[str] = None,
    ):
        super().__init__(name=f"MemoryWatchdog-{job_id or 'global'}", daemon=True)
        # Scale the floors down on small unified boards so they aren't
        # permanently below the soft limit.
        total_gb = memory_reader().system_total
        self.soft_free_gb = min(soft_free_gb, total_gb * 0.10) if total_gb else soft_free_gb
        self.hard_free_gb = min(hard_free_gb, total_gb * 0.04) if total_gb else hard_free_gb
        self.poll_seconds = poll_seconds
        self.on_soft_limit = on_soft_limit
        self.on_hard_limit = on_hard_limit
        self.log_path = log_path
        self.memory_reader = memory_reader
        self.job_id = job_id

        self._stop_event = threading.Event()
        self.soft_triggered = False
        self.hard_triggered = False
        self.last_sample: Optional[MemorySample] = None

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.check_once()
            except Exception as e:
                logger.debug(f"Memory watchdog sample failed: {e}")
            self._stop_event.wait(self.poll_seconds)

    def check_once(self) -> MemorySample:
        """Take one sample, log it, and fire escalations. Separated from
        run() so tests can drive the state machine synchronously."""
        sample = self.memory_reader()
        self.last_sample = sample
        self._write_forensic_line(sample)

        if sample.system_available < self.hard_free_gb and not self.hard_triggered:
            self.hard_triggered = True
            self.soft_triggered = True
            logger.critical(
                f"Memory watchdog: CRITICAL — {sample.system_available:.1f}GB free "
                f"(< {self.hard_free_gb:.1f}GB hard floor). {sample.format_line()}"
            )
            if self.on_hard_limit:
                self.on_hard_limit(sample)
        elif sample.system_available < self.soft_free_gb and not self.soft_triggered:
            self.soft_triggered = True
            logger.warning(
                f"Memory watchdog: pressure — {sample.system_available:.1f}GB free "
                f"(< {self.soft_free_gb:.1f}GB soft floor). {sample.format_line()}"
            )
            if self.on_soft_limit:
                self.on_soft_limit(sample)
        return sample

    def stop(self) -> None:
        self._stop_event.set()

    def _write_forensic_line(self, sample: MemorySample) -> None:
        """Append + fsync one line so the log survives a hard machine crash."""
        if self.log_path is None:
            return
        try:
            path = Path(self.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.stat().st_size > _FORENSIC_LOG_MAX_BYTES:
                path.replace(path.with_suffix(path.suffix + ".1"))
            prefix = f"[{self.job_id}] " if self.job_id else ""
            with open(path, "a") as f:
                f.write(prefix + sample.format_line() + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            logger.debug(f"Could not write memory forensic log: {e}")


class TrainingMemoryGuard:
    """
    Facade the training runners use. One instance per job:

        guard = TrainingMemoryGuard(job_id, notify=...)
        guard.install()             # in the training thread, before model load
        ...
        guard.set_trainer(trainer)  # once the trainer exists
        ...
        guard.shutdown()            # in the finally block

    Inert (all methods no-ops) when the system is not unified-memory or the
    guard is disabled in settings — callers never need to branch.
    """

    def __init__(
        self,
        job_id: str,
        notify: Optional[Callable[[str, str], None]] = None,
        settings: Any = None,
        memory_reader: Callable[[], MemorySample] = _default_memory_reader,
    ):
        self.job_id = job_id
        self.notify = notify  # notify(level, message) -> None
        self._settings = settings
        self._memory_reader = memory_reader
        self._trainer_lock = threading.Lock()
        self._trainer: Any = None
        self._training_thread: Optional[threading.Thread] = None
        self._watchdog: Optional[MemoryWatchdog] = None
        self.info: Optional[UnifiedMemoryInfo] = None
        self.active = False
        self.pressure_event: Optional[str] = None

    def _get_settings(self) -> Any:
        if self._settings is not None:
            return self._settings
        try:
            from config import get_settings
            self._settings = get_settings()
        except Exception:
            self._settings = None
        return self._settings

    def install(self, training_thread: Optional[threading.Thread] = None) -> "TrainingMemoryGuard":
        settings = self._get_settings()
        enabled = getattr(settings, "memory_guard_enabled", True) if settings else True
        if not enabled:
            logger.info("Memory guard disabled via MEMORY_GUARD_ENABLED=false")
            return self

        override = getattr(settings, "unified_memory", None) if settings else None
        self.info = detect_unified_memory(override=override)
        if not self.info.is_unified:
            logger.debug(f"Memory guard idle: {self.info.reason}")
            return self

        logger.info(
            f"🛡️ Unified-memory system detected ({self.info.reason}) — "
            "memory guard active for this job"
        )
        reserve_gb = getattr(settings, "memory_guard_reserve_gb", DEFAULT_RESERVE_GB) \
            if settings else DEFAULT_RESERVE_GB
        apply_memory_limits(self.info, reserve_gb=reserve_gb)

        self._training_thread = training_thread or threading.current_thread()

        log_path = None
        if not settings or getattr(settings, "memory_guard_log_enabled", True):
            data_dir = Path(getattr(settings, "data_dir", "./data")) if settings else Path("./data")
            log_path = data_dir / "memory_guard.log"

        self._watchdog = MemoryWatchdog(
            soft_free_gb=getattr(settings, "memory_guard_soft_free_gb", DEFAULT_SOFT_FREE_GB)
            if settings else DEFAULT_SOFT_FREE_GB,
            hard_free_gb=getattr(settings, "memory_guard_hard_free_gb", DEFAULT_HARD_FREE_GB)
            if settings else DEFAULT_HARD_FREE_GB,
            poll_seconds=getattr(settings, "memory_guard_poll_seconds", DEFAULT_POLL_SECONDS)
            if settings else DEFAULT_POLL_SECONDS,
            on_soft_limit=self._handle_soft_limit,
            on_hard_limit=self._handle_hard_limit,
            log_path=log_path,
            memory_reader=self._memory_reader,
            job_id=self.job_id,
        )
        self._watchdog.start()
        self.active = True
        return self

    def set_trainer(self, trainer: Any) -> None:
        with self._trainer_lock:
            self._trainer = trainer

    def shutdown(self) -> None:
        if self._watchdog is not None:
            self._watchdog.stop()
            self._watchdog = None
        self.set_trainer(None)
        self.active = False

    # ---- escalation handlers (called from the watchdog thread) ----

    def _handle_soft_limit(self, sample: MemorySample) -> None:
        message = (
            f"System memory pressure: only {sample.system_available:.1f}GB free on a "
            "unified-memory machine. Requesting graceful training stop to save a "
            "checkpoint before the OS runs out of RAM."
        )
        self.pressure_event = message
        self._notify("warning", message)
        with self._trainer_lock:
            trainer = self._trainer
        if trainer is not None:
            try:
                trainer.request_stop()
                logger.warning(f"Memory guard: graceful stop requested for job {self.job_id}")
                return
            except Exception as e:
                logger.error(f"Memory guard: request_stop failed: {e}")
        # No trainer yet (model/dataset still loading): nothing to stop
        # gracefully. The hard floor will abort the thread if this keeps
        # climbing; log so the escalation is traceable.
        logger.warning(
            f"Memory guard: soft limit hit before trainer exists for job {self.job_id} "
            "(load-time spike) — will hard-abort if pressure reaches the critical floor"
        )

    def _handle_hard_limit(self, sample: MemorySample) -> None:
        message = (
            f"CRITICAL memory pressure: {sample.system_available:.1f}GB free. "
            f"Aborting job {self.job_id} to keep the machine alive."
        )
        self.pressure_event = message
        self._notify("error", message)
        if _abort_thread(self._training_thread, MemoryPressureAbort):
            logger.critical(f"Memory guard: aborted training thread for job {self.job_id}")
        else:
            logger.critical(
                f"Memory guard: could not abort training thread for job {self.job_id} — "
                "if pressure continues the OS may become unresponsive"
            )

    def _notify(self, level: str, message: str) -> None:
        if self.notify is None:
            return
        try:
            self.notify(level, message)
        except Exception as e:
            logger.debug(f"Memory guard notify failed: {e}")
