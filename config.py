"""
Merlina Configuration Management

Centralized configuration using Pydantic BaseSettings.
Configuration can be set via:
1. .env file (recommended for secrets)
2. Environment variables
3. Defaults in this file
"""

import os
import logging
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from pydantic import PrivateAttr, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    Settings priority (highest to lowest):
    1. Environment variables
    2. .env file
    3. Default values
    """

    # ==========================================
    # Server Configuration
    # ==========================================
    app_name: str = "Merlina"
    host: str = "0.0.0.0"
    port: int = 8000
    domain: Optional[str] = None  # e.g., "https://merlina.example.com" for production

    # ==========================================
    # Directory Paths
    # ==========================================
    data_dir: Path = Path("./data")
    models_dir: Path = Path("./models")
    results_dir: Path = Path("./results")
    uploads_dir: Path = Path("./data/uploads")
    frontend_dir: Path = Path("./frontend")

    # ==========================================
    # Database
    # ==========================================
    database_path: str = "./data/jobs.db"

    # ==========================================
    # External Services (API Keys)
    # ==========================================
    wandb_api_key: Optional[str] = None
    hf_token: Optional[str] = None

    # ==========================================
    # Object storage tiers (S3-compatible)
    # ==========================================
    # Two role-named tiers with one consistent shape. Each is any S3 API
    # (self-hosted MinIO, Cloudflare R2, AWS, ...); only endpoint and keys
    # differ. Leave a tier unset to disable it. Credentials belong in the
    # server's .env, never in code or version control. See
    # docs/storage-tiers.md.
    #
    # internal: the private primary (typically LAN-only). Private data never
    #   leaves it: `merlina datasets promote` and `merlina storage sync` refuse
    #   anything marked private.
    storage_internal_endpoint: Optional[str] = None   # e.g. "http://minio.lan:9000"
    storage_internal_bucket: Optional[str] = None     # e.g. "datasets"
    storage_internal_access_key: Optional[str] = None
    storage_internal_secret_key: Optional[str] = None
    storage_internal_region: Optional[str] = None     # default "auto"
    # Optional admin key, used only for bucket-level operations (list/create/
    # delete buckets, browse non-default buckets). Falls back to the key above.
    storage_internal_admin_access_key: Optional[str] = None
    storage_internal_admin_secret_key: Optional[str] = None
    # external: a shared tier reachable off-LAN (e.g. from a rented GPU box).
    storage_external_endpoint: Optional[str] = None   # e.g. "https://<account>.r2.cloudflarestorage.com"
    storage_external_bucket: Optional[str] = None
    storage_external_access_key: Optional[str] = None
    storage_external_secret_key: Optional[str] = None
    storage_external_region: Optional[str] = None
    storage_external_admin_access_key: Optional[str] = None
    storage_external_admin_secret_key: Optional[str] = None

    # Which tier THIS process resolves swl:// datasets against: "internal"
    # (default) or "external" (a rented GPU box).
    dataset_tier: Optional[str] = None
    # Mirror "huggingface" sources into the internal tier on first use and
    # serve them from there afterwards.
    dataset_mirror: Optional[bool] = None

    # Where remote/hosted runs keep artifacts: "hf_hub" (default; private HF
    # repos), or a tier name ("external" / "internal"), which reuses that
    # tier's endpoint and credentials. Remote (RunPod) runs need "external":
    # the internal tier is LAN-only.
    artifact_store: Optional[str] = None
    artifact_bucket: Optional[str] = None   # default: the tier's bucket
    artifact_prefix: Optional[str] = None   # default: "merlina/runs"
    merlina_checkpoint_timeout_seconds: int = 120

    # ------------------------------------------------------------------
    # Deprecated aliases (still honoured; the names above win). Resolved by
    # _resolve_storage_aliases; a single warning lists any that are set.
    # The MERLINA_ARTIFACT_ACCESS_KEY / _SECRET_KEY / _SESSION_TOKEN names are
    # also the artifact *worker wire format*: the control plane forwards them
    # to remote workers, which read them here (see src/remote/artifacts.py).
    # ------------------------------------------------------------------
    s3_endpoint_url: Optional[str] = None
    s3_dataset_bucket: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    s3_region: Optional[str] = None
    r2_endpoint_url: Optional[str] = None
    r2_dataset_bucket: Optional[str] = None
    r2_access_key: Optional[str] = None
    r2_secret_key: Optional[str] = None
    r2_region: Optional[str] = None
    dataset_store: Optional[str] = None
    prefer_internal: Optional[bool] = None
    merlina_artifact_store: Optional[str] = None
    merlina_artifact_s3_bucket: Optional[str] = None
    merlina_artifact_s3_prefix: Optional[str] = None
    merlina_artifact_s3_endpoint: Optional[str] = None
    merlina_artifact_s3_region: Optional[str] = None
    merlina_artifact_access_key: Optional[str] = None
    merlina_artifact_secret_key: Optional[str] = None
    merlina_artifact_session_token: Optional[str] = None
    _deprecated_storage_names: list = PrivateAttr(default_factory=list)
    _conflicting_storage_names: list = PrivateAttr(default_factory=list)

    # ==========================================
    # Training Defaults
    # ==========================================
    default_lora_r: int = 64
    default_lora_alpha: int = 32
    default_learning_rate: float = 5e-6
    default_batch_size: int = 1
    default_gradient_accumulation_steps: int = 16
    default_num_epochs: int = 2

    # ==========================================
    # Job Queue Configuration
    # ==========================================
    max_concurrent_jobs: int = 1  # Maximum number of training jobs to run simultaneously

    # ==========================================
    # Remote Execution (rented compute)
    # ==========================================
    # API key for the RunPod provider (https://www.runpod.io/console/user/settings).
    # Remote jobs are provisioned on YOUR provider account — Merlina only
    # orchestrates. Leave unset to disable remote mode.
    runpod_api_key: Optional[str] = None
    # Container image the provisioned worker instances run. Must contain
    # Merlina (the worker entry is `python -m src.remote.worker_entry`).
    remote_worker_image: str = "ghcr.io/schneewolf-labs/merlina:latest"
    # Remote jobs don't occupy a local GPU, so they get their own queue.
    remote_max_concurrent_jobs: int = 2
    # Control-plane polling cadence against remote workers (seconds).
    remote_poll_interval_seconds: float = 10.0
    # How long a fresh instance may take to pull the image and come up.
    remote_boot_timeout_minutes: float = 30.0
    # After forwarding a stop, how long to wait for a graceful checkpoint
    # before terminating the instance anyway.
    remote_stop_grace_minutes: float = 15.0
    # How long worker status may stay unreachable (proxy 404/5xx, timeouts)
    # before a running stage is declared failed. Provider HTTP proxies blip.
    remote_status_error_tolerance_minutes: float = 10.0
    # Directories whose files remote `local_file` datasets may be staged from
    # (comma or os.pathsep separated). Empty refuses every remote local file.
    remote_local_data_roots: str = ""

    # ==========================================
    # System Configuration
    # ==========================================
    cuda_visible_devices: Optional[str] = None
    log_level: str = "INFO"

    # ==========================================
    # Unified-Memory Protection (DGX Spark / Grace-Blackwell)
    # ==========================================
    # On unified-memory machines (GB10 "DGX Spark", GH200, Jetson) the GPU
    # and system RAM share one pool, so a training memory spike can take
    # down the whole machine instead of raising a clean CUDA OOM. The
    # memory guard (src/memory_guard.py) caps the CUDA allocator and runs
    # a watchdog that stops training before the OS starves. It activates
    # automatically only when a unified-memory system is detected.
    memory_guard_enabled: bool = True
    unified_memory: str = "auto"          # auto | true | false (force detection result)
    memory_guard_reserve_gb: float = 12.0  # RAM kept off-limits to the CUDA allocator
    memory_guard_soft_free_gb: float = 8.0  # below this free RAM: graceful training stop
    memory_guard_hard_free_gb: float = 3.0  # below this free RAM: abort the job
    memory_guard_poll_seconds: float = 5.0
    memory_guard_log_enabled: bool = True   # fsync'd forensic log at data/memory_guard.log
    # Before a soft-floor stop, the watchdog releases the CUDA allocator's
    # idle cache (post-eval slack) and re-checks. If pressure comes back
    # within this window it stops for real instead of thrashing the cache.
    memory_guard_reclaim_cooldown_seconds: float = 60.0

    # Offline mode: never hit the HuggingFace Hub. Exports HF_HUB_OFFLINE /
    # TRANSFORMERS_OFFLINE so transformers/datasets/huggingface_hub resolve
    # everything from the local cache. Models and datasets must already be
    # downloaded (or referenced by local path).
    offline_mode: bool = False

    # ==========================================
    # Security & Limits
    # ==========================================
    cors_origins: List[str] = ["*"]
    max_upload_size_mb: int = 500

    # ==========================================
    # llama.cpp Integration (Optional)
    # ==========================================
    # Used for GGUF export / quantization and synthetic-data generation via
    # llama-server. Both are optional — features gracefully degrade when the
    # binaries cannot be located. See src/llama_cpp_resolver.py for the full
    # discovery precedence (explicit > env > PATH > vendored checkout).
    llama_cpp_dir: Optional[Path] = None       # Repo root (contains convert_hf_to_gguf.py)
    llama_cpp_bin_dir: Optional[Path] = None   # Directory of built binaries

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore unknown environment variables

    @model_validator(mode="after")
    def _resolve_storage_aliases(self):
        """Fold deprecated storage names into the role-named ones (new wins)."""
        used, conflicts = [], []
        for new, old in STORAGE_ALIASES:
            old_val = getattr(self, old)
            if old_val is None:
                continue
            if old.startswith("merlina_artifact_") and not self.merlina_artifact_store:
                # Bare credentials without MERLINA_ARTIFACT_STORE are the
                # worker wire format (see artifact_location), not a config.
                continue
            used.append(old)
            new_val = getattr(self, new)
            if new_val is None:
                setattr(self, new, old_val)
            elif new_val != old_val:
                conflicts.append(old)
        if self.merlina_artifact_store:
            used.append("merlina_artifact_store")
            for name in ("merlina_artifact_s3_bucket", "merlina_artifact_s3_endpoint",
                         "merlina_artifact_s3_region", "merlina_artifact_access_key",
                         "merlina_artifact_secret_key", "merlina_artifact_session_token"):
                if getattr(self, name) is not None:
                    used.append(name)
        self._deprecated_storage_names = used
        self._conflicting_storage_names = conflicts
        _warn_deprecated_storage_names(used, conflicts)
        return self

    def artifact_location(self) -> "ArtifactLocation":
        """The resolved artifact store (see :func:`resolve_artifact_location`)."""
        return resolve_artifact_location(self)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Apply offline mode before any HF library reads the environment
        if self.offline_mode:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            logger.info(
                "Offline mode enabled: HuggingFace Hub access disabled, "
                "models and datasets will be loaded from the local cache only"
            )
        # Apply and validate CUDA_VISIBLE_DEVICES if set
        if self.cuda_visible_devices:
            validation_result = validate_cuda_visible_devices(self.cuda_visible_devices)
            if validation_result["valid"]:
                os.environ['CUDA_VISIBLE_DEVICES'] = self.cuda_visible_devices
                if validation_result.get("warnings"):
                    for warning in validation_result["warnings"]:
                        logger.warning(warning)
            else:
                # Log error but don't crash - let training fail later with better context
                for error in validation_result.get("errors", []):
                    logger.error(error)


# ── storage settings: aliases and the resolved artifact location ─────────────
# (new canonical attribute, deprecated alias attribute). Env names are the
# upper-cased attribute names.
STORAGE_ALIASES = (
    ("storage_internal_endpoint", "s3_endpoint_url"),
    ("storage_internal_bucket", "s3_dataset_bucket"),
    ("storage_internal_access_key", "s3_access_key"),
    ("storage_internal_secret_key", "s3_secret_key"),
    ("storage_internal_region", "s3_region"),
    ("storage_external_endpoint", "r2_endpoint_url"),
    ("storage_external_bucket", "r2_dataset_bucket"),
    ("storage_external_access_key", "r2_access_key"),
    ("storage_external_secret_key", "r2_secret_key"),
    ("storage_external_region", "r2_region"),
    ("dataset_tier", "dataset_store"),
    ("dataset_mirror", "prefer_internal"),
    ("artifact_prefix", "merlina_artifact_s3_prefix"),
)
# Where the legacy MERLINA_ARTIFACT_* block maps (reported in the warning).
_LEGACY_ARTIFACT_NOTE = {
    "merlina_artifact_store": "ARTIFACT_STORE",
    "merlina_artifact_s3_bucket": "ARTIFACT_STORE=<tier> (+ ARTIFACT_BUCKET)",
    "merlina_artifact_s3_endpoint": "ARTIFACT_STORE=<tier>",
    "merlina_artifact_s3_region": "ARTIFACT_STORE=<tier>",
    "merlina_artifact_access_key": "ARTIFACT_STORE=<tier>",
    "merlina_artifact_secret_key": "ARTIFACT_STORE=<tier>",
    "merlina_artifact_session_token": "ARTIFACT_STORE=<tier>",
}
DEFAULT_ARTIFACT_PREFIX = "merlina/runs"
STORAGE_TIER_NAMES = ("internal", "external")
_DEPRECATION_WARNED = False


def _warn_deprecated_storage_names(used, conflicts) -> None:
    """One warning per process listing deprecated names (never values)."""
    global _DEPRECATION_WARNED
    if not used or _DEPRECATION_WARNED:
        return
    _DEPRECATION_WARNED = True
    mapping = dict((old, new) for new, old in STORAGE_ALIASES)
    parts = [f"{old.upper()} -> {(mapping[old].upper() if old in mapping else _LEGACY_ARTIFACT_NOTE.get(old, '?'))}"
             for old in dict.fromkeys(used)]
    msg = "Deprecated storage settings in use: " + "; ".join(parts) + "."
    if conflicts:
        msg += (" Both old and new names are set with different values for: "
                + ", ".join(c.upper() for c in conflicts) + " (the new names win).")
    msg += " Run `merlina config migrate-env` to rename them (dry run by default)."
    logger.warning(msg)


def _norm_endpoint(url) -> str:
    return (url or "").strip().rstrip("/").lower()


@dataclass(frozen=True)
class ArtifactLocation:
    """Where artifacts live, resolved from settings. ``kind`` is ``"hf_hub"``
    or ``"s3"``; ``tier`` names the storage tier whose endpoint and keys are
    used (``None`` for a legacy custom location)."""

    kind: str
    tier: Optional[str] = None
    bucket: Optional[str] = None
    prefix: str = DEFAULT_ARTIFACT_PREFIX
    endpoint_url: Optional[str] = None
    region: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    session_token: Optional[str] = None
    error: Optional[str] = None  # set when the configuration is unusable

    def __repr__(self) -> str:  # never print credentials
        return (f"ArtifactLocation(kind={self.kind!r}, tier={self.tier!r}, bucket={self.bucket!r}, "
                f"prefix={self.prefix!r}, error={self.error!r})")


def tier_settings(settings, tier: str) -> dict:
    """``{endpoint, bucket, access_key, secret_key, region, admin_*}`` for a tier."""
    g = lambda k: getattr(settings, f"storage_{tier}_{k}", None)  # noqa: E731
    return {"endpoint": g("endpoint"), "bucket": g("bucket"), "access_key": g("access_key"),
            "secret_key": g("secret_key"), "region": g("region") or "auto",
            "admin_access_key": g("admin_access_key"), "admin_secret_key": g("admin_secret_key")}


def _legacy_artifact_tier(settings) -> Optional[str]:
    """The tier a legacy MERLINA_ARTIFACT_STORE=s3 block points at: same
    endpoint and the same credentials (or none given), and no session token.
    ``None`` means a custom location (e.g. dedicated, prefix-scoped keys)."""
    if getattr(settings, "merlina_artifact_session_token", None):
        return None
    endpoint = _norm_endpoint(getattr(settings, "merlina_artifact_s3_endpoint", None))
    ak = getattr(settings, "merlina_artifact_access_key", None)
    sk = getattr(settings, "merlina_artifact_secret_key", None)
    for tier in ("external", "internal"):
        t = tier_settings(settings, tier)
        if not (t["endpoint"] and t["bucket"]) or _norm_endpoint(t["endpoint"]) != endpoint:
            continue
        if (ak is None and sk is None) or (ak == t["access_key"] and sk == t["secret_key"]):
            return tier
    return None


def resolve_artifact_location(settings) -> ArtifactLocation:
    """Resolve ``ARTIFACT_STORE`` (or the legacy ``MERLINA_ARTIFACT_*`` block)
    into an :class:`ArtifactLocation`. Works on any settings-like object."""
    prefix = (getattr(settings, "artifact_prefix", None)
              or getattr(settings, "merlina_artifact_s3_prefix", None) or DEFAULT_ARTIFACT_PREFIX)
    store = (getattr(settings, "artifact_store", None) or "").strip().lower()
    legacy = (getattr(settings, "merlina_artifact_store", None) or "").strip().lower()
    bucket_override = getattr(settings, "artifact_bucket", None)
    if not store and legacy == "s3":
        tier = _legacy_artifact_tier(settings)
        if tier is None:
            return ArtifactLocation(
                kind="s3", tier=None, bucket=getattr(settings, "merlina_artifact_s3_bucket", None),
                prefix=prefix, endpoint_url=getattr(settings, "merlina_artifact_s3_endpoint", None),
                region=getattr(settings, "merlina_artifact_s3_region", None),
                access_key=getattr(settings, "merlina_artifact_access_key", None),
                secret_key=getattr(settings, "merlina_artifact_secret_key", None),
                session_token=getattr(settings, "merlina_artifact_session_token", None),
                error=None if getattr(settings, "merlina_artifact_s3_bucket", None)
                else "MERLINA_ARTIFACT_S3_BUCKET is required for a custom S3 artifact location")
        store = tier
        legacy_bucket = getattr(settings, "merlina_artifact_s3_bucket", None)
        if not bucket_override and legacy_bucket and legacy_bucket != tier_settings(settings, tier)["bucket"]:
            bucket_override = legacy_bucket
    elif not store:
        store = legacy or "hf_hub"
    if store == "hf_hub":
        return ArtifactLocation(kind="hf_hub", prefix=prefix)
    if store not in STORAGE_TIER_NAMES:
        return ArtifactLocation(kind="invalid", prefix=prefix,
                                error="ARTIFACT_STORE must be hf_hub, external or internal")
    t = tier_settings(settings, store)
    bucket = bucket_override or t["bucket"]
    error = None
    if not t["endpoint"] or not bucket:
        error = (f"ARTIFACT_STORE={store} needs the {store} tier configured "
                 f"(STORAGE_{store.upper()}_ENDPOINT / _BUCKET / keys)")
    return ArtifactLocation(kind="s3", tier=store, bucket=bucket, prefix=prefix,
                            endpoint_url=t["endpoint"], region=t["region"],
                            access_key=t["access_key"], secret_key=t["secret_key"], error=error)


def artifact_credentials(settings) -> tuple:
    """``(access_key, secret_key, session_token)`` for the artifact store.

    The control plane uses the resolved location (a tier's keys, or a legacy
    custom block). A remote worker has no ARTIFACT_STORE; it receives the keys
    under the MERLINA_ARTIFACT_* wire names, which are read here as-is."""
    loc = resolve_artifact_location(settings)
    if loc.kind == "s3":
        return loc.access_key, loc.secret_key, loc.session_token
    return (getattr(settings, "merlina_artifact_access_key", None),
            getattr(settings, "merlina_artifact_secret_key", None),
            getattr(settings, "merlina_artifact_session_token", None))


def resolve_hf_token(token: Optional[str]) -> Optional[str]:
    """
    Resolve the HuggingFace token to use.

    Precedence:
    1. Explicit token passed in (e.g., from HTTP request)
    2. HF_TOKEN from settings (.env file / environment variables)

    This lets clients omit ``hf_token`` from requests when the server
    already has one configured via ``.env``, so secrets don't need to be
    sent over HTTP.
    """
    if token:
        return token
    return settings.hf_token or os.getenv("HF_TOKEN") or None


def resolve_wandb_key(key: Optional[str]) -> Optional[str]:
    """
    Resolve the Weights & Biases API key to use.

    Precedence:
    1. Explicit key passed in (e.g., from HTTP request)
    2. WANDB_API_KEY from settings (.env file / environment variables)
    """
    if key:
        return key
    return settings.wandb_api_key or os.getenv("WANDB_API_KEY") or None


def env_secret_status() -> dict:
    """
    Report which secrets are available from the server environment.

    Used by the frontend to show users whether they need to provide
    tokens in the UI or can rely on the server's ``.env`` configuration.
    The actual secret values are never returned.
    """
    return {
        "hf_token": bool(settings.hf_token or os.getenv("HF_TOKEN")),
        "wandb_api_key": bool(settings.wandb_api_key or os.getenv("WANDB_API_KEY")),
    }


def validate_cuda_visible_devices(cuda_str: str) -> dict:
    """
    Validate CUDA_VISIBLE_DEVICES string before applying it.

    Args:
        cuda_str: The CUDA_VISIBLE_DEVICES value (e.g., "0", "0,1", "GPU-uuid")

    Returns:
        dict with "valid" bool, and optional "errors" and "warnings" lists
    """
    result = {"valid": True, "errors": [], "warnings": []}

    if not cuda_str or cuda_str.strip() == "":
        return result

    # Check if CUDA is even available before validating device IDs
    try:
        import torch
        if not torch.cuda.is_available():
            result["warnings"].append(
                f"CUDA_VISIBLE_DEVICES='{cuda_str}' is set but CUDA is not available. "
                "This setting will have no effect."
            )
            return result

        actual_gpu_count = torch.cuda.device_count()
    except ImportError:
        result["warnings"].append(
            "PyTorch not available for CUDA validation. Skipping GPU index validation."
        )
        return result
    except Exception as e:
        result["warnings"].append(f"Could not check CUDA availability: {e}")
        return result

    # Parse the CUDA_VISIBLE_DEVICES value
    # It can be: "0", "0,1,2", "GPU-<uuid>", etc.
    devices = [d.strip() for d in cuda_str.split(",")]

    for device in devices:
        # Skip UUID-based device specifications
        if device.startswith("GPU-") or device.startswith("MIG-"):
            continue

        # Try to parse as integer
        try:
            device_id = int(device)

            # Note: We validate against actual_gpu_count BEFORE setting CUDA_VISIBLE_DEVICES
            # because once it's set, torch.cuda.device_count() returns the filtered count
            if device_id < 0:
                result["valid"] = False
                result["errors"].append(
                    f"Invalid GPU index '{device_id}': GPU indices must be non-negative."
                )
            elif device_id >= actual_gpu_count:
                result["valid"] = False
                result["errors"].append(
                    f"Invalid GPU index '{device_id}': Only {actual_gpu_count} GPU(s) available (indices 0-{actual_gpu_count - 1})."
                )

        except ValueError:
            result["valid"] = False
            result["errors"].append(
                f"Invalid CUDA_VISIBLE_DEVICES value '{device}': "
                "Expected integer GPU index or GPU-<uuid> format."
            )

    return result


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings
