#!/usr/bin/env python3
"""
Regression tests: heavy work must never run on the API event loop.

A single slow or hung model load used to freeze /health and every other
endpoint, because the load ran synchronously inside an `async def` endpoint
on the event loop. These tests pin the fix:

- Endpoints that do heavy synchronous work (model/tokenizer/dataset loads,
  diffusion generation, disk walks/deletes) are plain `def` endpoints, so
  FastAPI runs them in its threadpool instead of on the event loop.
- The /train submission path never imports the heavy training stack on the
  event loop — that import happens on the job-queue worker thread.
- /health stays responsive while a slow model load is in flight.

Runnable standalone (`python tests/test_event_loop_blocking.py`) or under
pytest (ML dependencies are mocked either way — no torch/GPU required).
"""

import os
import sys
import time
import asyncio
import threading
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Mock ML dependencies (mirrors tests/conftest.py for standalone runs) ─────

if 'torch' not in sys.modules:
    class FakeTensor:
        pass

    class FakeModule:
        pass

    mock_torch = MagicMock()
    mock_torch.__spec__ = MagicMock()
    mock_torch.Tensor = FakeTensor
    mock_torch.nn = MagicMock()
    mock_torch.nn.Module = FakeModule
    mock_torch.cuda.is_available.return_value = False
    mock_torch.cuda.device_count.return_value = 0
    mock_torch.cuda.empty_cache = Mock()
    mock_torch.bfloat16 = "bfloat16"
    mock_torch.float16 = "float16"
    sys.modules['torch'] = mock_torch
    sys.modules['torch.nn'] = mock_torch.nn

if 'transformers' not in sys.modules:
    class FakeTokenizerBase:
        pass

    mock_transformers = MagicMock()
    mock_transformers.PreTrainedTokenizerBase = FakeTokenizerBase
    sys.modules['transformers'] = mock_transformers

for _module in [
    'trl', 'trl.experimental', 'trl.experimental.orpo',
    'peft', 'accelerate', 'bitsandbytes', 'wandb', 'psutil', 'pynvml',
    'grimoire', 'grimoire.losses', 'grimoire.data',
    'artemis_vlm',
    'atelier', 'atelier.adapters', 'atelier.losses', 'atelier.data',
    'diffusers',
]:
    if _module not in sys.modules:
        sys.modules[_module] = MagicMock()

# Pure-python deps that CI installs but a minimal environment may lack.
for _module in ['datasets', 'datasets.arrow_dataset', 'datasets.load',
                'pyarrow', 'huggingface_hub']:
    try:
        __import__(_module)
    except ImportError:
        sys.modules[_module] = MagicMock()

import merlina  # noqa: E402


# ── 1. Heavy endpoints must be sync (threadpool), light ones async ───────────

# Every endpoint that can block for a long time on network, disk, or model
# deserialization. FastAPI runs plain `def` endpoints in its threadpool, so
# a slow call here can never freeze the event loop.
HEAVY_SYNC_ENDPOINTS = [
    "preload_model_tokenizer",     # POST /model/preload
    "detect_model_layers",         # POST /model/layers
    "load_inference_model",        # POST /inference/load
    "unload_inference_model",      # POST /inference/unload
    "preview_dataset",             # POST /dataset/preview
    "preview_formatted_dataset",   # POST /dataset/preview-formatted
    "get_dataset_stats",           # POST /dataset/stats
    "get_dataset_columns",         # POST /dataset/columns
    "diffusion_generate",          # POST /diffusion/generate
    "disk_analysis",               # GET  /disk/analysis
    "disk_cleanup",                # POST /disk/cleanup
    "disk_models_delete",          # POST /disk/models/delete
    "disk_gguf_delete",            # POST /disk/artifacts/gguf/delete
    "disk_wandb_clear",            # POST /disk/artifacts/wandb/clear
]


def test_heavy_endpoints_are_sync():
    """Heavy endpoints must be `def`, not `async def`, so they run in the
    threadpool. An `async def` here reintroduces the API-wide freeze."""
    for name in HEAVY_SYNC_ENDPOINTS:
        fn = getattr(merlina, name)
        assert not asyncio.iscoroutinefunction(fn), (
            f"{name} is `async def` — its blocking body would run on the "
            f"event loop and freeze every endpoint. Make it a plain `def` "
            f"(or move the blocking work into asyncio.to_thread)."
        )


def test_health_endpoint_stays_on_event_loop():
    """/health must stay async + cheap so it answers even when the
    threadpool is busy with slow loads."""
    assert asyncio.iscoroutinefunction(merlina.health_check)


# ── 2. /train must not import the training stack on the event loop ──────────

def test_training_callback_defers_heavy_import():
    """_make_training_callback runs on the event loop inside /train.
    Importing src.training_runner there (grimoire, peft, wandb, ...) can
    stall the API for tens of seconds on first call — the import must be
    deferred to the queue worker thread that invokes the callback."""
    saved_module = sys.modules.pop('src.training_runner', None)
    try:
        callback = merlina._make_training_callback(None)

        assert callable(callback)
        assert 'src.training_runner' not in sys.modules, (
            "_make_training_callback imported src.training_runner eagerly — "
            "that import must happen inside the callback (worker thread), "
            "never on the API event loop."
        )
    finally:
        # Put the original module object back so other tests that patch
        # src.training_runner attributes keep targeting the same instance.
        if saved_module is not None:
            sys.modules['src.training_runner'] = saved_module


# ── 3. Live check: /health answers while a model load is in flight ──────────

def test_health_responsive_during_slow_model_load():
    """Fire a /model/preload whose tokenizer load blocks until released,
    and verify /health still answers immediately. Before the fix, the load
    ran on the event loop and /health hung for its full duration."""
    from fastapi.testclient import TestClient

    load_started = threading.Event()
    release_load = threading.Event()

    def slow_tokenizer_load(*args, **kwargs):
        load_started.set()
        release_load.wait(timeout=10)  # hung-download stand-in
        tok = MagicMock()
        tok.vocab_size = 32000
        tok.model_max_length = 4096
        tok.chat_template = None
        tok.pad_token = "<pad>"
        tok.eos_token = "</s>"
        tok.bos_token = "<s>"
        return tok

    fake_auto_tokenizer = MagicMock()
    fake_auto_tokenizer.from_pretrained = Mock(side_effect=slow_tokenizer_load)

    with patch.object(merlina, "AutoTokenizer", fake_auto_tokenizer):
        with TestClient(merlina.app) as client:
            preload_result = {}

            def do_preload():
                preload_result["response"] = client.post(
                    "/model/preload",
                    json={"model_name": "test/slow-model-event-loop-check"},
                )

            preload_thread = threading.Thread(target=do_preload, daemon=True)
            preload_thread.start()
            assert load_started.wait(timeout=5), "preload never reached the tokenizer load"

            # The load is now blocked mid-flight. /health must still answer.
            t0 = time.monotonic()
            health = client.get("/health")
            elapsed = time.monotonic() - t0

            release_load.set()
            preload_thread.join(timeout=10)

            assert health.status_code == 200
            assert elapsed < 2.0, (
                f"/health took {elapsed:.1f}s while a model load was in "
                f"flight — the load is blocking the event loop again."
            )
            assert preload_result["response"].status_code == 200


if __name__ == "__main__":
    print("=" * 60)
    print("Event-loop blocking regression tests")
    print("=" * 60)
    failures = 0
    for test in [
        test_heavy_endpoints_are_sync,
        test_health_endpoint_stays_on_event_loop,
        test_training_callback_defers_heavy_import,
        test_health_responsive_during_slow_model_load,
    ]:
        try:
            test()
            print(f"✅ {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"❌ {test.__name__}: {exc}")
    print("=" * 60)
    if failures:
        print(f"❌ {failures} test(s) failed")
        sys.exit(1)
    print("✅ All tests passed!")
