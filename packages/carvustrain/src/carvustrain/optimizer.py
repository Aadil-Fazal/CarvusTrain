"""Optimizer algorithms including Adam, AdamW, SGD, RMSprop, and AdaGrad with gradient clipping and weight decay."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union


class Optimizer:
    """Abstract base optimizer class."""

    def __init__(
        self,
        params: Optional[List[Any]] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip: Optional[float] = 1.0,
    ) -> None:
        self.params = params or []
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.step_count = 0

    def step(self) -> None:
        """Perform optimization update step."""
        self.step_count += 1
        self._update_params()

    def _update_params(self) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Clear parameter gradients."""
        try:
            import torch

            for p in self.params:
                if isinstance(p, torch.Tensor) and p.grad is not None:
                    p.grad.zero_()
        except ImportError:
            pass

    def clip_gradients(self, max_norm: float = 1.0) -> float:
        """Clip gradient norms to prevent exploding gradients."""
        try:
            import torch

            if self.params and isinstance(self.params[0], torch.Tensor):
                return float(torch.nn.utils.clip_grad_norm_(self.params, max_norm))
        except ImportError:
            pass
        return 0.0

    def state_dict(self) -> Dict[str, Any]:
        """Serialize optimizer state."""
        return {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore optimizer state."""
        self.lr = state.get("lr", self.lr)
        self.weight_decay = state.get("weight_decay", self.weight_decay)
        self.grad_clip = state.get("grad_clip", self.grad_clip)
        self.step_count = state.get("step_count", self.step_count)


class Adam(Optimizer):
    """Adam Optimizer."""

    def __init__(
        self,
        params: Optional[List[Any]] = None,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        grad_clip: Optional[float] = 1.0,
    ) -> None:
        super().__init__(params, lr, weight_decay, grad_clip)
        self.beta1, self.beta2 = betas
        self.eps = eps

    def _update_params(self) -> None:
        try:
            import torch

            if self.params and isinstance(self.params[0], torch.Tensor):
                if self.grad_clip:
                    self.clip_gradients(self.grad_clip)

                for p in self.params:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if self.weight_decay != 0:
                        grad = grad.add(p.data, alpha=self.weight_decay)

                    state = getattr(p, "_adam_state", {})
                    if not state:
                        state["m"] = torch.zeros_like(p.data)
                        state["v"] = torch.zeros_like(p.data)
                        p._adam_state = state  # type: ignore

                    m, v = state["m"], state["v"]
                    m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
                    v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

                    bias_correction1 = 1 - self.beta1**self.step_count
                    bias_correction2 = 1 - self.beta2**self.step_count

                    step_size = self.lr * (math.sqrt(bias_correction2) / bias_correction1)
                    denom = v.sqrt().add_(self.eps)
                    p.data.addcdiv_(m, denom, value=-step_size)
        except ImportError:
            pass


class AdamW(Adam):
    """AdamW Optimizer with decoupled weight decay."""

    def _update_params(self) -> None:
        try:
            import torch

            if self.params and isinstance(self.params[0], torch.Tensor):
                if self.grad_clip:
                    self.clip_gradients(self.grad_clip)

                for p in self.params:
                    if p.grad is None:
                        continue
                    grad = p.grad.data

                    # Decoupled weight decay
                    if self.weight_decay != 0:
                        p.data.mul_(1 - self.lr * self.weight_decay)

                    state = getattr(p, "_adamw_state", {})
                    if not state:
                        state["m"] = torch.zeros_like(p.data)
                        state["v"] = torch.zeros_like(p.data)
                        p._adamw_state = state  # type: ignore

                    m, v = state["m"], state["v"]
                    m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
                    v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

                    bias_correction1 = 1 - self.beta1**self.step_count
                    bias_correction2 = 1 - self.beta2**self.step_count

                    step_size = self.lr * (math.sqrt(bias_correction2) / bias_correction1)
                    denom = v.sqrt().add_(self.eps)
                    p.data.addcdiv_(m, denom, value=-step_size)
        except ImportError:
            pass


class SGD(Optimizer):
    """Stochastic Gradient Descent with optional momentum."""

    def __init__(
        self,
        params: Optional[List[Any]] = None,
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        grad_clip: Optional[float] = 1.0,
    ) -> None:
        super().__init__(params, lr, weight_decay, grad_clip)
        self.momentum = momentum

    def _update_params(self) -> None:
        try:
            import torch

            if self.params and isinstance(self.params[0], torch.Tensor):
                if self.grad_clip:
                    self.clip_gradients(self.grad_clip)

                for p in self.params:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if self.weight_decay != 0:
                        grad = grad.add(p.data, alpha=self.weight_decay)

                    if self.momentum != 0:
                        state = getattr(p, "_sgd_state", {})
                        if not state:
                            state["velocity"] = torch.zeros_like(p.data)
                            p._sgd_state = state  # type: ignore
                        velocity = state["velocity"]
                        velocity.mul_(self.momentum).add_(grad)
                        grad = velocity

                    p.data.add_(grad, alpha=-self.lr)
        except ImportError:
            pass


class RMSprop(Optimizer):
    """RMSprop Optimizer."""

    def __init__(
        self,
        params: Optional[List[Any]] = None,
        lr: float = 1e-3,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        grad_clip: Optional[float] = 1.0,
    ) -> None:
        super().__init__(params, lr, weight_decay, grad_clip)
        self.alpha = alpha
        self.eps = eps

    def _update_params(self) -> None:
        try:
            import torch

            if self.params and isinstance(self.params[0], torch.Tensor):
                for p in self.params:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    state = getattr(p, "_rmsprop_state", {})
                    if not state:
                        state["square_avg"] = torch.zeros_like(p.data)
                        p._rmsprop_state = state  # type: ignore
                    sq_avg = state["square_avg"]
                    sq_avg.mul_(self.alpha).addcmul_(grad, grad, value=1 - self.alpha)
                    avg = sq_avg.sqrt().add_(self.eps)
                    p.data.addcdiv_(grad, avg, value=-self.lr)
        except ImportError:
            pass


class AdaGrad(Optimizer):
    """AdaGrad Optimizer."""

    def __init__(
        self,
        params: Optional[List[Any]] = None,
        lr: float = 1e-2,
        eps: float = 1e-10,
        weight_decay: float = 0.0,
        grad_clip: Optional[float] = 1.0,
    ) -> None:
        super().__init__(params, lr, weight_decay, grad_clip)
        self.eps = eps

    def _update_params(self) -> None:
        try:
            import torch

            if self.params and isinstance(self.params[0], torch.Tensor):
                for p in self.params:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    state = getattr(p, "_adagrad_state", {})
                    if not state:
                        state["sum"] = torch.zeros_like(p.data)
                        p._adagrad_state = state  # type: ignore
                    sum_sq = state["sum"]
                    sum_sq.addcmul_(grad, grad, value=1.0)
                    std = sum_sq.sqrt().add_(self.eps)
                    p.data.addcdiv_(grad, std, value=-self.lr)
        except ImportError:
            pass
