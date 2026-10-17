"""
Datatypes shared across the remote-execution package.

Pure dataclasses with zero ML or HTTP dependencies so that sizing,
planning, and orchestration logic stays unit-testable everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GpuOffer:
    """A rentable GPU type as advertised by a compute provider."""
    gpu_type_id: str                  # provider's identifier, e.g. "NVIDIA H200"
    display_name: str
    vram_gb: float
    price_per_hr_secure: Optional[float] = None
    price_per_hr_community: Optional[float] = None
    max_gpu_count: int = 8
    available: bool = True

    def price_per_hr(self, cloud_type: str = "secure") -> Optional[float]:
        # No fallback to the other cloud's price: a GPU with no secure price isn't sold on
        # secure cloud, and borrowing the community price let sizing pick an instance that
        # provisioning could never get.
        price = self.price_per_hr_community if cloud_type == "community" else self.price_per_hr_secure
        return price or None


@dataclass
class InstanceSpec:
    """Everything a provider needs to create an instance."""
    name: str
    image: str
    gpu_type_id: Optional[str] = None     # None = CPU-only instance
    gpu_count: int = 1
    container_disk_gb: int = 50
    volume_gb: int = 0
    volume_mount_path: str = "/workspace"
    cloud_type: str = "secure"            # "secure" | "community"
    env: Dict[str, str] = field(default_factory=dict)
    http_ports: List[int] = field(default_factory=lambda: [8000])
    docker_start_cmd: Optional[List[str]] = None
    interruptible: bool = False           # spot instances (resume not yet supported)


@dataclass
class RemoteInstance:
    """A provisioned instance, as tracked by the orchestrator."""
    instance_id: str
    provider: str
    status: str = "pending"               # pending | running | exited | terminated | unknown
    cost_per_hr: Optional[float] = None
    gpu_type_id: Optional[str] = None
    gpu_count: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_running(self) -> bool:
        return self.status == "running"


@dataclass
class ModelSpecs:
    """
    Architecture facts about a base model, read from its config.json.

    ``total_params_b`` / ``active_params_b`` are estimates derived from the
    config (MoE-aware); ``weight_bytes_total`` is exact when the repo
    publishes a safetensors index.
    """
    model_id: str
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    vocab_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    tie_word_embeddings: bool = False
    architectures: List[str] = field(default_factory=list)

    # MoE fields (None / 0 for dense models)
    num_routed_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    num_shared_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    first_k_dense_layers: int = 0

    # Attention/KV facts used by serve-time KV-cache sizing. MLA models
    # (DeepSeek-V2/V3-style) cache a compressed latent of width
    # kv_lora_rank + qk_rope_head_dim instead of per-head K/V.
    kv_lora_rank: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None

    total_params_b: Optional[float] = None
    active_params_b: Optional[float] = None
    weight_bytes_total: Optional[int] = None   # from safetensors index metadata

    @property
    def is_moe(self) -> bool:
        return bool(self.num_routed_experts and self.num_routed_experts > 1)


@dataclass
class SizingDecision:
    """The advisor's pick: which instance shape a stage should run on."""
    gpu_type_id: str
    gpu_count: int
    cloud_type: str
    vram_required_gb: float
    disk_required_gb: float
    est_cost_per_hr: Optional[float]
    # "single_gpu" fits on one GPU; "model_parallel" shards layers across
    # GPUs via device_map=auto (single process); "ddp" replicates per GPU.
    parallelism: str = "single_gpu"
    rationale: List[str] = field(default_factory=list)

    @property
    def multi_gpu_strategy(self) -> str:
        """Map parallelism onto Merlina's TrainingConfig.multi_gpu_strategy."""
        if self.parallelism == "model_parallel":
            return "single"
        if self.parallelism == "ddp":
            return "ddp"
        return "single"


@dataclass
class StagePlan:
    """One stage of a piecemeal remote run."""
    name: str                              # "train" | "merge" | "serve"
    target: str                            # "remote" | "local" | "skip"
    sizing: Optional[SizingDecision] = None
    artifacts_in: List[str] = field(default_factory=list)
    artifacts_out: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class RemotePlan:
    """Full plan for a remote run: stages + the facts that produced them."""
    job_id: Optional[str]
    model_specs: Optional[ModelSpecs]
    stages: List[StagePlan] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def stage(self, name: str) -> Optional[StagePlan]:
        for s in self.stages:
            if s.name == name:
                return s
        return None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe summary for API responses and job metadata."""
        from dataclasses import asdict
        return asdict(self)


@dataclass
class ServePlan:
    """
    Dry-run plan for a remote inference session: what gets served, on which
    instance, at what price, and when it is forcibly torn down. Built by
    :func:`src.remote.inference.build_serve_plan`; the same object drives
    the real session, so what the owner approves is what runs.
    """
    model_source: Dict[str, Any]           # {"hf_repo", "revision"} | {"store_model"} | {"job_id", "artifact"}
    model_id: str                          # what vLLM loads (HF id, "store:<name>", or an adapter's base)
    revision: Optional[str]
    served_model_name: str                 # name clients pass as "model"
    base_served_model_name: Optional[str]  # adapter sessions also serve the bare base
    lora: bool                             # serve base + LoRA adapter (vLLM --enable-lora)
    max_lora_rank: Optional[int]
    artifact: Optional[str]                # "merged" | "adapter" | a store_model name when pulled from a store
    artifact_store: Optional[Dict[str, Any]]
    model_specs: Optional[ModelSpecs]
    sizing: SizingDecision
    tensor_parallel_size: int
    max_model_len: int
    max_num_seqs: int
    gpu_memory_utilization: float
    vram_breakdown: Dict[str, float]
    max_runtime_hours: float
    max_total_cost_usd: Optional[float]
    idle_timeout_minutes: Optional[float]
    effective_max_hours: float             # min(runtime cap, total-cost cap / $/hr)
    deadline_reason: str                   # "max_runtime_hours" | "max_total_cost_usd"
    est_max_total_cost_usd: Optional[float]
    worker_image: str
    vllm_args: List[str] = field(default_factory=list)
    vllm_command: List[str] = field(default_factory=list)   # display only; never holds the key
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)
