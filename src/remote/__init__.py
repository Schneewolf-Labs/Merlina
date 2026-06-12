"""
Remote execution for Merlina — run training stages on rented compute.

This package lets a locally-running Merlina act as a control plane that
provisions cloud GPU instances (RunPod first), runs pipeline stages on
them, and hands artifacts between stages through a durable store.

The pipeline is deliberately piecemeal: each stage (train, merge) can run
on a different machine — even a different provider — because stages only
communicate through the artifact store. A 1T-class MoE QLoRA run can
train on an 8×H200 pod, ship a few-GB adapter to a private HF repo, and
merge (or not) somewhere with lots of RAM and disk instead of GPUs.

Everything in this package is importable without torch/transformers; ML
imports happen lazily inside stage execution (which runs on the remote
worker, where the full stack is installed).
"""

from .spec import (
    GpuOffer,
    InstanceSpec,
    RemoteInstance,
    ModelSpecs,
    SizingDecision,
    StagePlan,
    RemotePlan,
)

__all__ = [
    "GpuOffer",
    "InstanceSpec",
    "RemoteInstance",
    "ModelSpecs",
    "SizingDecision",
    "StagePlan",
    "RemotePlan",
]
