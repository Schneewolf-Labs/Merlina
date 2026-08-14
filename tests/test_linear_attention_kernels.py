"""A hybrid model training on the reference path should say so.

Qwen3.5/3.6/3.8 interleave linear-attention layers with full-attention ones. Those layers have a
fused training path that is off unless both optional kernels are installed, and **nothing in the
loss curve reveals which one ran** — a job that takes seven times longer than necessary looks
completely normal from the outside.

Measured on ReAligned-Qwen3.5-4B, 4096 tokens, batch 1, gradient checkpointing on, steady state
after warm-up:

    torch fallback         13.67 s/step
    fla + causal_conv1d     1.84 s/step     7.4x

Equivalence checked on identical input: cosine 0.99996 in float64.

These tests exist because the first version of this check referenced `config` inside a method that
does not receive it. The resulting NameError was swallowed by a `try/except Exception`, so the
check silently never ran — which is exactly the failure it was written to prevent.
"""

from types import SimpleNamespace
from unittest.mock import patch

from src.preflight_checks import PreflightValidator
from src.vram_estimator import ModelProfile


def checker() -> PreflightValidator:
    c = PreflightValidator()
    c.warnings = []
    c.errors = []
    return c


def cfg(model: str = "Qwen/Qwen3.5-4B") -> SimpleNamespace:
    return SimpleNamespace(base_model=model, hf_token=None)


def hybrid(n: int = 24) -> ModelProfile:
    return ModelProfile(model_id="x", hidden_size=2560, num_hidden_layers=32,
                        linear_attention_layers=n)


def dense() -> ModelProfile:
    return ModelProfile(model_id="x", hidden_size=2560, num_hidden_layers=32,
                        linear_attention_layers=0)


def _missing(*names: str):
    """Fail imports for the named modules only.

    A blanket `__import__` mock also breaks the method's own internal imports, so the check never
    reaches the code under test and the test passes for the wrong reason.
    """
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name in names:
            raise ImportError(f"no {name}")
        return real(name, *a, **k)

    return fake


def test_warns_when_kernels_missing_on_a_hybrid_model():
    c = checker()
    with patch("src.vram_estimator.get_model_profile", return_value=hybrid()), \
         patch("builtins.__import__", side_effect=_missing("fla", "causal_conv1d")):
        c._check_linear_attention_kernels(cfg())
    assert c.warnings, "hybrid model with no kernels produced no warning"
    assert "linear-attention" in c.warnings[0]


def test_names_the_missing_module_and_how_to_install_it():
    c = checker()
    with patch("src.vram_estimator.get_model_profile", return_value=hybrid()), \
         patch("builtins.__import__", side_effect=_missing("fla")):
        c._check_linear_attention_kernels(cfg())
    assert c.warnings
    assert "fla" in c.warnings[0]
    assert "pip install" in c.warnings[0]


def test_silent_on_a_dense_model():
    """A model with no linear-attention layers has nothing to install."""
    c = checker()
    with patch("src.vram_estimator.get_model_profile", return_value=dense()), \
         patch("builtins.__import__", side_effect=_missing("fla", "causal_conv1d")):
        c._check_linear_attention_kernels(cfg("meta-llama/Llama-3-8B"))
    assert c.warnings == []


def test_an_unresolvable_model_is_not_an_error():
    """Preflight must not fail a job because the profile could not be fetched."""
    c = checker()
    with patch("src.vram_estimator.get_model_profile", side_effect=RuntimeError("offline")):
        c._check_linear_attention_kernels(cfg())
    assert c.warnings == []
    assert c.errors == []


def test_profile_counts_linear_attention_layers_from_config():
    """The count comes from the model's own layer_types, not a name heuristic."""
    from src.vram_estimator import _profile_from_config_dict

    cfg_dict = {
        "text_config": {
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "layer_types": ["linear_attention"] * 24 + ["full_attention"] * 8,
        }
    }
    p = _profile_from_config_dict("x", cfg_dict, "hub_config")
    assert p.linear_attention_layers == 24
