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
        if cloud_type == "community":
            return self.price_per_hr_community or self.price_per_hr_secure
        return self.price_per_hr_secure or self.price_per_hr_community


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
    name: str                              # "train" | "merge"
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
