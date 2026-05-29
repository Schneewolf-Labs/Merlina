"""Tests for `_cleanup_training_resources` and `_release_cuda_tensors`.

These guard the post-failure GPU cleanup path that runs in the `finally`
block of every training runner (text / VLM / diffusion). The bug they
prevent: when a job dies during model load or trainer init, lingering
references (peft hooks, accelerator state, partially-constructed
trainers) keep the model alive past `del model`, so `gc.collect()` and
`empty_cache()` can't release VRAM — and the next queued job OOMs.

Run from anywhere — works under the mocked torch in tests/conftest.py
because we check reassignment via sentinels rather than tensor shape.
"""
import gc
import sys
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Allow `python tests/test_cleanup.py` standalone runs (CLAUDE.md pattern).
sys.path.insert(0, "/home/python/AI/GrimoireProject/Merlina")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

class _FakeCudaParam:
    """Parameter-like stand-in whose .device reports CUDA without needing a GPU.

    `_data_sentinel` lets a test verify that `_release_cuda_tensors`
    actually reassigned `.data` (by checking identity drift), without
    depending on the resulting tensor's shape — which is helpful since
    tests/conftest.py mocks `torch` and a mocked `torch.empty(...)` is
    a MagicMock, not a real zero-numel tensor.
    """

    def __init__(self):
        self._data_sentinel = object()
        self.data = self._data_sentinel
        self._grad_sentinel = object()
        self.grad = self._grad_sentinel
        # Spoof the device — _release_cuda_tensors reads `.device.type`.
        self.device = SimpleNamespace(type="cuda")


class _FakeCpuParam:
    """Like _FakeCudaParam but reports CPU — must NOT be mutated by cleanup."""

    def __init__(self):
        self._data_sentinel = object()
        self.data = self._data_sentinel
        self.grad = None
        self.device = SimpleNamespace(type="cpu")


class _FakeModule:
    """Module-like stand-in with parameters() / buffers() iterators."""

    def __init__(self, params=None, buffers=None):
        self._params = list(params) if params is not None else [_FakeCudaParam() for _ in range(3)]
        self._buffers = list(buffers) if buffers is not None else [_FakeCudaParam() for _ in range(2)]

    def parameters(self, recurse=True):
        return iter(self._params)

    def buffers(self, recurse=True):
        return iter(self._buffers)


def _reassigned(param) -> bool:
    """True iff `_release_cuda_tensors` replaced .data away from the sentinel."""
    return param.data is not param._data_sentinel


# ---------------------------------------------------------------------
# _release_cuda_tensors
# ---------------------------------------------------------------------

class TestReleaseCudaTensors:
    def test_none_safe(self):
        from src.training_runner import _release_cuda_tensors
        # Plain object with no parameters()/buffers() — must not raise.
        _release_cuda_tensors(SimpleNamespace())

    def test_reassigns_cuda_param_data_and_nulls_grad(self):
        from src.training_runner import _release_cuda_tensors
        mod = _FakeModule()
        _release_cuda_tensors(mod)
        for p in mod._params:
            assert _reassigned(p), "CUDA param.data must be reassigned"
            assert p.grad is None, "CUDA param.grad must be nulled"
        for b in mod._buffers:
            assert _reassigned(b), "CUDA buffer.data must be reassigned"

    def test_leaves_cpu_params_alone(self):
        from src.training_runner import _release_cuda_tensors
        cpu_params = [_FakeCpuParam() for _ in range(3)]
        mod = _FakeModule(params=cpu_params, buffers=[])
        _release_cuda_tensors(mod)
        for p in cpu_params:
            assert not _reassigned(p), (
                "CPU param.data must NOT be touched — release is CUDA-only"
            )

    def test_mixed_cpu_and_cuda(self):
        """Mixed model: only the CUDA tensors get released."""
        from src.training_runner import _release_cuda_tensors
        cuda_params = [_FakeCudaParam() for _ in range(2)]
        cpu_params = [_FakeCpuParam() for _ in range(2)]
        mod = _FakeModule(params=cuda_params + cpu_params, buffers=[])
        _release_cuda_tensors(mod)
        for p in cuda_params:
            assert _reassigned(p)
        for p in cpu_params:
            assert not _reassigned(p)

    def test_survives_broken_parameter_iter(self):
        """A partially-built model with a broken parameters() must not raise."""
        from src.training_runner import _release_cuda_tensors

        class Broken:
            def parameters(self, recurse=True):
                raise RuntimeError("partial model is busted")
            def buffers(self, recurse=True):
                raise RuntimeError("also busted")

        # Must swallow — cleanup is best-effort and never masks the
        # original training failure that triggered it.
        _release_cuda_tensors(Broken())

    def test_survives_individual_param_failure(self):
        """One broken param shouldn't stop the rest from being released."""
        from src.training_runner import _release_cuda_tensors

        class ExplodingParam:
            @property
            def device(self):
                raise RuntimeError("boom")

        good_a = _FakeCudaParam()
        good_b = _FakeCudaParam()
        bad = ExplodingParam()

        class Mixed:
            def parameters(self, recurse=True):
                return iter([good_a, bad, good_b])
            def buffers(self, recurse=True):
                return iter([])

        _release_cuda_tensors(Mixed())
        assert _reassigned(good_a), "good param before the bad one must be released"
        assert _reassigned(good_b), "good param after the bad one must be released"


# ---------------------------------------------------------------------
# _cleanup_training_resources
# ---------------------------------------------------------------------

class TestCleanupTrainingResources:
    def test_both_none(self):
        """Most pessimistic path: training died before model or trainer existed."""
        from src.training_runner import _cleanup_training_resources
        _cleanup_training_resources(None, None)

    def test_model_only(self):
        from src.training_runner import _cleanup_training_resources
        mod = _FakeModule()
        _cleanup_training_resources(mod, None)
        for p in mod._params:
            assert _reassigned(p)

    def test_trainer_only_nulls_all_known_ref_attrs(self):
        from src.training_runner import _cleanup_training_resources
        trainer = SimpleNamespace(
            optimizer=MagicMock(),
            lr_scheduler=MagicMock(),
            scheduler=MagicMock(),
            accelerator=MagicMock(),
            _accelerator=MagicMock(),
            train_dataloader=MagicMock(),
            eval_dataloader=MagicMock(),
            model=MagicMock(),
            ref_model=MagicMock(),
            _model=MagicMock(),
            callbacks=[MagicMock()],
        )
        _cleanup_training_resources(None, trainer)
        # Every known ref-holding attr must be nulled — that's what
        # actually lets gc reclaim the model the trainer was holding.
        for attr in ("optimizer", "lr_scheduler", "scheduler", "accelerator",
                     "_accelerator", "train_dataloader", "eval_dataloader",
                     "model", "ref_model", "_model", "callbacks"):
            assert getattr(trainer, attr) is None, f"trainer.{attr} should be None"

    def test_partial_trainer_missing_attrs_is_ok(self):
        """A trainer that died inside __init__ may not have all attrs set yet."""
        from src.training_runner import _cleanup_training_resources
        trainer = SimpleNamespace(model=MagicMock())  # only one of the canonical attrs
        _cleanup_training_resources(None, trainer)
        assert trainer.model is None

    def test_diffusion_adapter_submodules_walked(self):
        """Atelier adapters wrap multiple sub-models; each must be released.

        Regression test for the original bug — a diffusion adapter is a
        plain wrapper (NOT an nn.Module) with `.transformer`, `.vae`,
        `.text_encoder` sub-models. Without walking these the adapter's
        sub-models would keep their CUDA params alive after cleanup.
        """
        from src.training_runner import _cleanup_training_resources

        transformer = _FakeModule()
        vae = _FakeModule(params=[_FakeCudaParam()], buffers=[])
        text_encoder = _FakeModule(params=[_FakeCudaParam()], buffers=[])

        # Adapter exposes the sub-models by name — like Atelier's real adapters.
        adapter = SimpleNamespace(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
        )

        _cleanup_training_resources(adapter, None)
        for sub in (transformer, vae, text_encoder):
            for p in sub._params:
                assert _reassigned(p), (
                    "diffusion adapter sub-module params must be released — "
                    "this is the bug the Atelier-side fix targets"
                )

    def test_cleanup_drops_trainer_strong_ref_to_model(self):
        """End-to-end: after cleanup, a model only held by trainer is collectable.

        This is the actual leak the bug report described — a
        partially-constructed trainer pinned the model past `del model`,
        and gc couldn't reclaim it, so VRAM stayed allocated.
        """
        from src.training_runner import _cleanup_training_resources

        class TinyModel:  # weakref-supporting (regular Python class is fine)
            pass

        model = TinyModel()
        trainer = SimpleNamespace(model=model, optimizer=None)
        ref = weakref.ref(model)

        # Caller passes the model only via the trainer's reference.
        _cleanup_training_resources(None, trainer)
        del model  # drop caller's local
        gc.collect()

        assert ref() is None, (
            "model should be garbage-collected once trainer.model is "
            "nulled and the caller's local goes out of scope — this is "
            "the exact failure mode the cleanup fix addresses"
        )

    def test_idempotent(self):
        """Workers may call cleanup more than once on shutdown / error paths."""
        from src.training_runner import _cleanup_training_resources
        mod = _FakeModule()
        trainer = SimpleNamespace(model=MagicMock(), optimizer=MagicMock())
        _cleanup_training_resources(mod, trainer)
        # Second call: model params are already reassigned, trainer attrs are None.
        # Must not raise.
        _cleanup_training_resources(mod, trainer)

    def test_handles_adapter_self_reference(self):
        """If adapter.model is the adapter itself, don't recurse infinitely."""
        from src.training_runner import _cleanup_training_resources

        adapter = _FakeModule()
        adapter.model = adapter  # cyclic reference — must be detected
        adapter.transformer = None
        adapter.vae = None
        adapter.text_encoder = None

        # The `sub is not model` guard in _cleanup_training_resources
        # prevents infinite recursion. Must complete and release params.
        _cleanup_training_resources(adapter, None)
        for p in adapter._params:
            assert _reassigned(p)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
