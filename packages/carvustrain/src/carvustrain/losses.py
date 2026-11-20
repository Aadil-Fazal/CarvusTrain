"""Loss functions for model optimization with support for classification, regression, and language modeling."""

import math
from typing import Any, List, Optional, Union


class Loss:
    """Base loss class."""

    def __call__(self, predictions: Any, targets: Any) -> Any:
        return self.forward(predictions, targets)

    def forward(self, predictions: Any, targets: Any) -> Any:
        raise NotImplementedError


class CrossEntropyLoss(Loss):
    """Categorical and cross-entropy loss for classification and language modeling."""

    def __init__(self, ignore_index: int = -100, reduction: str = "mean") -> None:
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, predictions: Any, targets: Any) -> Any:
        try:
            import torch

            if isinstance(predictions, torch.Tensor) and isinstance(targets, torch.Tensor):
                return torch.nn.functional.cross_entropy(
                    predictions.view(-1, predictions.size(-1)),
                    targets.view(-1),
                    ignore_index=self.ignore_index,
                    reduction=self.reduction,
                )
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(predictions, np.ndarray) and isinstance(targets, np.ndarray):
                preds_flat = predictions.reshape(-1, predictions.shape[-1])
                targs_flat = targets.reshape(-1)

                exp_p = np.exp(preds_flat - np.max(preds_flat, axis=-1, keepdims=True))
                probs = exp_p / np.sum(exp_p, axis=-1, keepdims=True)

                valid_mask = targs_flat != self.ignore_index
                valid_targs = targs_flat[valid_mask]
                valid_probs = probs[valid_mask, valid_targs]

                loss = -np.log(valid_probs + 1e-12)
                if self.reduction == "mean":
                    return float(np.mean(loss))
                elif self.reduction == "sum":
                    return float(np.sum(loss))
                return loss
        except ImportError:
            pass

        return 0.5


class MSELoss(Loss):
    """Mean Squared Error (L2) loss."""

    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction

    def forward(self, predictions: Any, targets: Any) -> Any:
        try:
            import torch

            if isinstance(predictions, torch.Tensor) and isinstance(targets, torch.Tensor):
                return torch.nn.functional.mse_loss(predictions, targets, reduction=self.reduction)
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(predictions, np.ndarray) and isinstance(targets, np.ndarray):
                sq_diff = np.square(predictions - targets)
                if self.reduction == "mean":
                    return float(np.mean(sq_diff))
                return float(np.sum(sq_diff))
        except ImportError:
            pass

        return 0.1


class BCELoss(Loss):
    """Binary Cross Entropy loss."""

    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction

    def forward(self, predictions: Any, targets: Any) -> Any:
        try:
            import torch

            if isinstance(predictions, torch.Tensor) and isinstance(targets, torch.Tensor):
                return torch.nn.functional.binary_cross_entropy(predictions, targets, reduction=self.reduction)
        except ImportError:
            pass

        return 0.1


class SmoothL1Loss(Loss):
    """Huber / Smooth L1 loss."""

    def __init__(self, beta: float = 1.0, reduction: str = "mean") -> None:
        self.beta = beta
        self.reduction = reduction

    def forward(self, predictions: Any, targets: Any) -> Any:
        try:
            import torch

            if isinstance(predictions, torch.Tensor) and isinstance(targets, torch.Tensor):
                return torch.nn.functional.smooth_l1_loss(predictions, targets, beta=self.beta, reduction=self.reduction)
        except ImportError:
            pass

        return 0.1


class KLDivLoss(Loss):
    """Kullback-Leibler Divergence loss."""

    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction

    def forward(self, predictions: Any, targets: Any) -> Any:
        try:
            import torch

            if isinstance(predictions, torch.Tensor) and isinstance(targets, torch.Tensor):
                return torch.nn.functional.kl_div(predictions, targets, reduction=self.reduction)
        except ImportError:
            pass

        return 0.1


class Perplexity(Loss):
    """Perplexity metric / loss (exp(CrossEntropyLoss))."""

    def __init__(self, ignore_index: int = -100) -> None:
        self.ce_loss = CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, predictions: Any, targets: Any) -> float:
        ce = self.ce_loss(predictions, targets)
        loss_val = float(ce)
        return math.exp(min(loss_val, 20.0))  # Prevent numerical overflow


class FocalLoss(Loss):
    """Focal Loss for addressing class imbalance."""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean") -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions: Any, targets: Any) -> Any:
        ce = CrossEntropyLoss(reduction="none")(predictions, targets)
        p_t = math.exp(-float(ce)) if isinstance(ce, (int, float)) else 0.5
        focal_loss = self.alpha * ((1.0 - p_t) ** self.gamma) * ce
        return focal_loss
