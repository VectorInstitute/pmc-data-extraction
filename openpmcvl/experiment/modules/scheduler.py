"""Load LR Scheduler from open_clip library."""

import math
from typing import List

from torch.optim.lr_scheduler import LRScheduler, _warn_get_lr_called_within_step
from torch.optim.optimizer import Optimizer


class CosineAnnealingWarmupLR(LRScheduler):
    """Copied from pytorch, and modified to add warmup steps."""

    def __init__(
        self,
        optimizer: Optimizer,
        t_max: int,
        warmup_length: int = 0,
        eta_min: float = 0,
        last_epoch: int = -1,
        verbose: str = "deprecated",
    ) -> None:
        self.t_max = t_max
        self.warmup_length = warmup_length
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """Get new lr values for all parameter groups."""
        _warn_get_lr_called_within_step(self)

        if self.last_epoch < self.warmup_length:
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_length
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        step = self.last_epoch - self.warmup_length
        total_steps = self.t_max - self.warmup_length
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos((step) * math.pi / total_steps))
            / 2
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
        ]
