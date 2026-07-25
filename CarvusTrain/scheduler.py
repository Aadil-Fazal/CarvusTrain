"""Learning rate schedulers for dynamic learning rate adjustment during training."""

import math
from typing import Any, Dict, List, Optional, Union

from .optimizer import Optimizer


class LRScheduler:
    """Base class for learning rate schedulers."""

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.last_epoch = last_epoch

    def step(self, metrics: Optional[float] = None) -> float:
        """Step the scheduler and update optimizer learning rate."""
        self.last_epoch += 1
        new_lr = self.get_lr(metrics)
        self.optimizer.lr = new_lr
        return new_lr

    def get_lr(self, metrics: Optional[float] = None) -> float:
        """Compute learning rate for current step."""
        raise NotImplementedError


class StepLR(LRScheduler):
    """Decays the learning rate by gamma every step_size epochs."""

    def __init__(self, optimizer: Optimizer, step_size: int = 10, gamma: float = 0.1, last_epoch: int = -1) -> None:
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self, metrics: Optional[float] = None) -> float:
        return self.base_lr * (self.gamma ** ((self.last_epoch + 1) // self.step_size))


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate schedule."""

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0, last_epoch: int = -1) -> None:
        self.T_max = max(1, T_max)
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self, metrics: Optional[float] = None) -> float:
        return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2


class LinearWarmupLR(LRScheduler):
    """Linear warmup scheduler scaling LR from 0 up to base_lr over warmup_steps."""

    def __init__(self, optimizer: Optimizer, warmup_steps: int = 100, last_epoch: int = -1) -> None:
        self.warmup_steps = max(1, warmup_steps)
        super().__init__(optimizer, last_epoch)

    def get_lr(self, metrics: Optional[float] = None) -> float:
        if self.last_epoch < self.warmup_steps:
            return self.base_lr * (self.last_epoch + 1) / self.warmup_steps
        return self.base_lr


class ExponentialLR(LRScheduler):
    """Exponential learning rate decay (LR = base_lr * gamma^epoch)."""

    def __init__(self, optimizer: Optimizer, gamma: float = 0.95, last_epoch: int = -1) -> None:
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self, metrics: Optional[float] = None) -> float:
        return self.base_lr * (self.gamma**self.last_epoch)


class ReduceLROnPlateau(LRScheduler):
    """Reduce learning rate when validation loss/metric has stopped improving."""

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 5,
        threshold: float = 1e-4,
        cooldown: int = 0,
        min_lr: float = 1e-7,
    ) -> None:
        super().__init__(optimizer)
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.best = float("inf") if mode == "min" else float("-inf")
        self.num_bad_epochs = 0
        self.cooldown_counter = 0

    def get_lr(self, metrics: Optional[float] = None) -> float:
        if metrics is None:
            return self.optimizer.lr

        current = float(metrics)
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return self.optimizer.lr

        is_better = (current < self.best - self.threshold) if self.mode == "min" else (current > self.best + self.threshold)
        if is_better:
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.num_bad_epochs = 0
            self.cooldown_counter = self.cooldown
            new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
            return new_lr

        return self.optimizer.lr
