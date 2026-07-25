"""Activation functions for neural networks with NumPy and PyTorch support."""

import math
from typing import Any, List, Union


class Activation:
    """Base activation class."""

    def __call__(self, x: Any) -> Any:
        return self.forward(x)

    def forward(self, x: Any) -> Any:
        raise NotImplementedError


class ReLU(Activation):
    """Rectified Linear Unit activation."""

    def forward(self, x: Any) -> Any:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                return torch.relu(x)
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(x, np.ndarray):
                return np.maximum(0, x)
        except ImportError:
            pass

        if isinstance(x, (list, tuple)):
            return [max(0.0, float(v)) for v in x]
        return max(0.0, float(x))


class GELU(Activation):
    """Gaussian Error Linear Unit activation."""

    def forward(self, x: Any) -> Any:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                return torch.nn.functional.gelu(x)
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(x, np.ndarray):
                return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        except ImportError:
            pass

        # Scalar fallback
        v = float(x)
        return 0.5 * v * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * (v**3))))


class SiLU(Activation):
    """Sigmoid Linear Unit (Swish) activation."""

    def forward(self, x: Any) -> Any:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                return torch.nn.functional.silu(x)
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(x, np.ndarray):
                return x / (1.0 + np.exp(-x))
        except ImportError:
            pass

        v = float(x)
        return v / (1.0 + math.exp(-v))


class Softmax(Activation):
    """Softmax activation along specified dimension."""

    def __init__(self, dim: int = -1) -> None:
        self.dim = dim

    def forward(self, x: Any) -> Any:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                return torch.softmax(x, dim=self.dim)
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(x, np.ndarray):
                exp_x = np.exp(x - np.max(x, axis=self.dim, keepdims=True))
                return exp_x / np.sum(exp_x, axis=self.dim, keepdims=True)
        except ImportError:
            pass

        if isinstance(x, (list, tuple)):
            max_v = max(x)
            exps = [math.exp(v - max_v) for v in x]
            sum_exps = sum(exps)
            return [e / sum_exps for e in exps]
        return 1.0


class Sigmoid(Activation):
    """Sigmoid activation."""

    def forward(self, x: Any) -> Any:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                return torch.sigmoid(x)
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(x, np.ndarray):
                return 1.0 / (1.0 + np.exp(-x))
        except ImportError:
            pass

        return 1.0 / (1.0 + math.exp(-float(x)))


class Tanh(Activation):
    """Hyperbolic tangent activation."""

    def forward(self, x: Any) -> Any:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                return torch.tanh(x)
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(x, np.ndarray):
                return np.tanh(x)
        except ImportError:
            pass

        return math.tanh(float(x))


class LeakyReLU(Activation):
    """Leaky Rectified Linear Unit activation."""

    def __init__(self, negative_slope: float = 0.01) -> None:
        self.negative_slope = negative_slope

    def forward(self, x: Any) -> Any:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                return torch.nn.functional.leaky_relu(x, negative_slope=self.negative_slope)
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(x, np.ndarray):
                return np.where(x > 0, x, x * self.negative_slope)
        except ImportError:
            pass

        v = float(x)
        return v if v > 0 else v * self.negative_slope
