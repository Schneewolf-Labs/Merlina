"""
Muon optimizer support for Merlina.

Provides a GrimoireTrainer subclass that adds Muon (MomentUm Orthogonalized
by Newton-Schulz) as an optimizer option.  Muon applies orthogonalized momentum
to 2D+ weight matrices and falls back to AdamW for 1D parameters (biases,
layer norms).

Reference: https://kellerjordan.github.io/posts/muon/
"""

import logging

import torch
from grimoire import GrimoireTrainer

logger = logging.getLogger(__name__)


class MuonGrimoireTrainer(GrimoireTrainer):
    """GrimoireTrainer with Muon optimizer support."""

    def __init__(self, *args, muon_momentum: float = 0.95, **kwargs):
        self._muon_momentum = muon_momentum
        super().__init__(*args, **kwargs)

    def _create_optimizer(self, model):
        if self.config.optimizer != "muon":
            return super()._create_optimizer(model)

        from muon import SingleDeviceMuonWithAuxAdam

        muon_params = []
        adam_params = []
        for _name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                muon_params.append(param)
            else:
                adam_params.append(param)

        lr = self.config.learning_rate
        wd = self.config.weight_decay

        logger.info(
            f"Muon optimizer: {len(muon_params)} Muon params (2D+), "
            f"{len(adam_params)} AdamW params (1D), "
            f"lr={lr}, momentum={self._muon_momentum}"
        )

        # If no 2D params exist, fall back to plain AdamW
        if not muon_params:
            logger.warning("No 2D+ parameters found for Muon — falling back to AdamW")
            return super()._create_optimizer(model)

        param_groups = [
            dict(
                params=muon_params,
                lr=lr,
                momentum=self._muon_momentum,
                weight_decay=wd,
                use_muon=True,
            ),
        ]

        if adam_params:
            param_groups.append(
                dict(
                    params=adam_params,
                    lr=lr,
                    betas=(0.9, 0.95),
                    eps=1e-10,
                    weight_decay=0.0,
                    use_muon=False,
                )
            )

        return SingleDeviceMuonWithAuxAdam(param_groups)
