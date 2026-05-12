"""
Warmup + Cosine Annealing LR Scheduler.

Single Responsibility: Learning rate scheduling with linear warmup
followed by cosine decay (Noam-style, standard for Conformer training).
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineScheduler(LambdaLR):
    """Linear warmup → cosine annealing learning rate schedule.

    Parameters
    ----------
    optimizer : Optimizer
        PyTorch optimizer.
    warmup_steps : int
        Number of linear warmup steps.
    total_steps : int
        Total training steps (warmup + decay).
    min_lr_ratio : float
        Minimum LR as fraction of peak LR (default 0.01).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.01,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(
                1, total_steps - warmup_steps
            )
            return max(
                min_lr_ratio,
                0.5 * (1.0 + math.cos(math.pi * progress)),
            )

        super().__init__(optimizer, lr_lambda)

    @classmethod
    def from_config(
        cls, cfg: dict, optimizer: Optimizer, total_steps: int
    ) -> "WarmupCosineScheduler":
        peak_lr = cfg.get("learning_rate", 0.001)
        min_lr = cfg.get("min_lr", 0.00001)
        min_lr_ratio = min_lr / peak_lr if peak_lr > 0 else 0.01
        return cls(
            optimizer=optimizer,
            warmup_steps=cfg.get("warmup_steps", 1000),
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        )
