"""Neural network layers including Transformer components, Attention, Conv, RNN, Embeddings, and Normalizations."""

import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union


class Layer:
    """Abstract base class for all neural network layers in CarvusTrain."""

    def __init__(self) -> None:
        self.training: bool = True
        self._parameters: List[Any] = []

    def __call__(self, x: Any, **kwargs: Any) -> Any:
        return self.forward(x, **kwargs)

    def forward(self, x: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def parameters(self) -> List[Any]:
        """Return all trainable parameters."""
        return self._parameters

    def eval(self) -> "Layer":
        self.training = False
        return self

    def train(self) -> "Layer":
        self.training = True
        return self


class Dense(Layer):
    """Fully connected linear layer (y = xW + b)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Xavier / Glorot uniform initialization
        limit = math.sqrt(6.0 / (in_features + out_features))
        self.weight: List[List[float]] = [
            [random.uniform(-limit, limit) for _ in range(out_features)]
            for _ in range(in_features)
        ]
        self.bias: Optional[List[float]] = [0.0] * out_features if bias else None

    def forward(self, x: Any, **kwargs: Any) -> Any:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                w = torch.tensor(self.weight, dtype=x.dtype, device=x.device)
                b = torch.tensor(self.bias, dtype=x.dtype, device=x.device) if self.bias else None
                return torch.matmul(x, w) + (b if b is not None else 0)
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(x, np.ndarray):
                w = np.array(self.weight)
                b = np.array(self.bias) if self.bias is not None else 0
                return np.matmul(x, w) + b
        except ImportError:
            pass

        # Python scalar/list fallback
        if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
            out = []
            for vec in x:
                row = []
                for col_idx in range(self.out_features):
                    val = sum(vec[i] * self.weight[i][col_idx] for i in range(self.in_features))
                    if self.bias:
                        val += self.bias[col_idx]
                    row.append(val)
                out.append(row)
            return out
        return [0.0] * self.out_features


Linear = Dense


class Conv2D(Layer):
    """2D Convolutional layer for image feature extraction.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (filters).
        kernel_size: Size of the convolution kernel (int or tuple).
        stride: Stride of the convolution (default: 1).
        padding: Zero-padding added to both sides (default: 0).
        bias: If True, adds a learnable bias (default: True).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.use_bias = bias

        # Kaiming / He uniform initialization
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        limit = math.sqrt(6.0 / fan_in) if fan_in > 0 else 0.1
        self.weight: List[List[List[List[float]]]] = [
            [
                [
                    [random.uniform(-limit, limit) for _ in range(self.kernel_size[1])]
                    for _ in range(self.kernel_size[0])
                ]
                for _ in range(in_channels)
            ]
            for _ in range(out_channels)
        ]
        self.bias_param: Optional[List[float]] = [0.0] * out_channels if bias else None

    def forward(self, x: Any, **kwargs: Any) -> Any:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                w = torch.tensor(self.weight, dtype=x.dtype, device=x.device)
                b = torch.tensor(self.bias_param, dtype=x.dtype, device=x.device) if self.bias_param else None
                return torch.nn.functional.conv2d(x, w, b, stride=self.stride, padding=self.padding)
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(x, np.ndarray):
                return self._conv2d_numpy(x, np)
        except ImportError:
            pass

        return x

    def _conv2d_numpy(self, x: Any, np: Any) -> Any:
        """Pure numpy Conv2D implementation."""
        if x.ndim == 3:
            x = x[None, ...]  # Add batch dimension

        batch, c_in, h_in, w_in = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        # Apply padding
        if ph > 0 or pw > 0:
            x = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")

        h_out = (h_in + 2 * ph - kh) // sh + 1
        w_out = (w_in + 2 * pw - kw) // sw + 1

        weight = np.array(self.weight)
        output = np.zeros((batch, self.out_channels, h_out, w_out))

        for b in range(batch):
            for oc in range(self.out_channels):
                for i in range(h_out):
                    for j in range(w_out):
                        region = x[b, :, i * sh : i * sh + kh, j * sw : j * sw + kw]
                        output[b, oc, i, j] = np.sum(region * weight[oc])
                if self.bias_param:
                    output[b, oc] += self.bias_param[oc]

        return output


class Embedding(Layer):
    """Token embedding layer mapping token IDs to dense vectors."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        scale = 1.0 / math.sqrt(embedding_dim)
        self.weight: List[List[float]] = [
            [random.uniform(-scale, scale) for _ in range(embedding_dim)]
            for _ in range(num_embeddings)
        ]

    def forward(self, x: Any, **kwargs: Any) -> Any:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                w = torch.tensor(self.weight, dtype=torch.float32, device=x.device)
                return torch.nn.functional.embedding(x, w)
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(x, np.ndarray):
                w = np.array(self.weight)
                return w[x]
        except ImportError:
            pass

        if isinstance(x, (list, tuple)):
            if isinstance(x[0], list):
                return [[self.weight[idx % self.num_embeddings] for idx in seq] for seq in x]
            return [self.weight[idx % self.num_embeddings] for idx in x]
        return self.weight[int(x) % self.num_embeddings]


class LayerNorm(Layer):
    """Layer Normalization."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = [1.0] * normalized_shape
        self.beta = [0.0] * normalized_shape

    def forward(self, x: Any, **kwargs: Any) -> Any:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                g = torch.tensor(self.gamma, dtype=x.dtype, device=x.device)
                b = torch.tensor(self.beta, dtype=x.dtype, device=x.device)
                return torch.nn.functional.layer_norm(x, (self.normalized_shape,), g, b, self.eps)
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(x, np.ndarray):
                mean = np.mean(x, axis=-1, keepdims=True)
                var = np.var(x, axis=-1, keepdims=True)
                norm = (x - mean) / np.sqrt(var + self.eps)
                return np.array(self.gamma) * norm + np.array(self.beta)
        except ImportError:
            pass

        return x


class BatchNorm(Layer):
    """Batch Normalization layer.

    Args:
        num_features: Number of features (channels) in the input.
        eps: Value added for numerical stability.
        momentum: Running mean/variance momentum factor.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = [1.0] * num_features
        self.beta = [0.0] * num_features
        self.running_mean = [0.0] * num_features
        self.running_var = [1.0] * num_features

    def forward(self, x: Any, **kwargs: Any) -> Any:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                g = torch.tensor(self.gamma, dtype=x.dtype, device=x.device)
                b = torch.tensor(self.beta, dtype=x.dtype, device=x.device)
                rm = torch.tensor(self.running_mean, dtype=x.dtype, device=x.device)
                rv = torch.tensor(self.running_var, dtype=x.dtype, device=x.device)
                return torch.nn.functional.batch_norm(x, rm, rv, g, b, self.training, self.momentum, self.eps)
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(x, np.ndarray):
                if self.training:
                    mean = np.mean(x, axis=0)
                    var = np.var(x, axis=0)
                    for i in range(self.num_features):
                        self.running_mean[i] = (1 - self.momentum) * self.running_mean[i] + self.momentum * float(
                            mean[i] if i < len(mean) else 0
                        )
                        self.running_var[i] = (1 - self.momentum) * self.running_var[i] + self.momentum * float(
                            var[i] if i < len(var) else 1
                        )
                else:
                    mean = np.array(self.running_mean)
                    var = np.array(self.running_var)
                norm = (x - mean) / np.sqrt(var + self.eps)
                return np.array(self.gamma) * norm + np.array(self.beta)
        except ImportError:
            pass

        return x


class RMSNorm(Layer):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = [1.0] * dim

    def forward(self, x: Any, **kwargs: Any) -> Any:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                w = torch.tensor(self.weight, dtype=x.dtype, device=x.device)
                variance = x.pow(2).mean(-1, keepdim=True)
                return x * torch.rsqrt(variance + self.eps) * w
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(x, np.ndarray):
                rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + self.eps)
                return (x / rms) * np.array(self.weight)
        except ImportError:
            pass

        return x


class PositionalEncoding(Layer):
    """Sinusoidal positional encoding layer for sequence representations."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Precompute encodings
        self.pe: List[List[float]] = []
        for pos in range(max_len):
            row = []
            for i in range(d_model):
                if i % 2 == 0:
                    val = math.sin(pos / (10000.0 ** (i / d_model)))
                else:
                    val = math.cos(pos / (10000.0 ** ((i - 1) / d_model)))
                row.append(val)
            self.pe.append(row)

    def forward(self, x: Any, **kwargs: Any) -> Any:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                seq_len = x.size(1) if x.dim() > 1 else x.size(0)
                pe_tensor = torch.tensor(self.pe[:seq_len], dtype=x.dtype, device=x.device)
                return x + pe_tensor
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(x, np.ndarray):
                seq_len = x.shape[1] if x.ndim > 1 else x.shape[0]
                pe_arr = np.array(self.pe[:seq_len])
                return x + pe_arr
        except ImportError:
            pass

        return x


class MultiHeadAttention(Layer):
    """Multi-Head Self-Attention layer."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_rate = dropout

        self.q_proj = Dense(embed_dim, embed_dim)
        self.k_proj = Dense(embed_dim, embed_dim)
        self.v_proj = Dense(embed_dim, embed_dim)
        self.out_proj = Dense(embed_dim, embed_dim)

    def forward(self, x: Any, mask: Optional[Any] = None, **kwargs: Any) -> Any:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
                attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                if mask is not None:
                    attn = attn.masked_fill(mask == 0, -1e9)
                attn_weights = torch.softmax(attn, dim=-1)
                output = torch.matmul(attn_weights, v)
                return self.out_proj(output)
        except ImportError:
            pass

        # NumPy/Python fallback
        return x


class FeedForward(Layer):
    """Transformer Feed-Forward Network (FFN)."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = Dense(embed_dim, hidden_dim)
        self.fc2 = Dense(hidden_dim, embed_dim)

    def forward(self, x: Any, **kwargs: Any) -> Any:
        h = self.fc1(x)
        try:
            import torch

            if isinstance(h, torch.Tensor):
                h = torch.nn.functional.gelu(h)
        except ImportError:
            pass
        return self.fc2(h)


class TransformerBlock(Layer):
    """Standard Transformer Encoder layer block."""

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln1 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, hidden_dim, dropout)
        self.ln2 = LayerNorm(embed_dim)

    def forward(self, x: Any, mask: Optional[Any] = None, **kwargs: Any) -> Any:
        # Self-Attention + Residual
        attn_out = self.attention(self.ln1(x), mask=mask)
        x = x + attn_out if hasattr(x, "__add__") else x

        # FFN + Residual
        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out if hasattr(x, "__add__") else x
        return x


class TransformerDecoderBlock(Layer):
    """Transformer Decoder block with self-attention, cross-attention, and FFN.

    Args:
        embed_dim: Model embedding dimension.
        num_heads: Number of attention heads.
        hidden_dim: Feed-forward hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)
        self.ln3 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, hidden_dim, dropout)

    def forward(self, x: Any, encoder_output: Optional[Any] = None, mask: Optional[Any] = None, **kwargs: Any) -> Any:
        # Masked self-attention + residual
        attn_out = self.self_attention(self.ln1(x), mask=mask)
        x = x + attn_out if hasattr(x, "__add__") else x

        # Cross-attention + residual
        if encoder_output is not None:
            cross_out = self.cross_attention(self.ln2(x))
            x = x + cross_out if hasattr(x, "__add__") else x

        # FFN + residual
        ffn_out = self.ffn(self.ln3(x))
        x = x + ffn_out if hasattr(x, "__add__") else x
        return x


class RNNCell(Layer):
    """Vanilla RNN cell: h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b).

    Args:
        input_size: Number of expected input features.
        hidden_size: Number of features in the hidden state.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        limit_ih = math.sqrt(6.0 / (input_size + hidden_size))
        limit_hh = math.sqrt(6.0 / (hidden_size + hidden_size))
        self.w_ih = [[random.uniform(-limit_ih, limit_ih) for _ in range(hidden_size)] for _ in range(input_size)]
        self.w_hh = [[random.uniform(-limit_hh, limit_hh) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.bias = [0.0] * hidden_size

    def forward(self, x: Any, h: Optional[Any] = None, **kwargs: Any) -> Any:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                w_ih = torch.tensor(self.w_ih, dtype=x.dtype, device=x.device)
                w_hh = torch.tensor(self.w_hh, dtype=x.dtype, device=x.device)
                b = torch.tensor(self.bias, dtype=x.dtype, device=x.device)
                if h is None:
                    h = torch.zeros(x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
                return torch.tanh(x @ w_ih + h @ w_hh + b)
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(x, np.ndarray):
                w_ih = np.array(self.w_ih)
                w_hh = np.array(self.w_hh)
                b = np.array(self.bias)
                if h is None:
                    h = np.zeros((x.shape[0], self.hidden_size) if x.ndim > 1 else (self.hidden_size,))
                return np.tanh(x @ w_ih + h @ w_hh + b)
        except ImportError:
            pass

        return [0.0] * self.hidden_size


class RNN(Layer):
    """Multi-step Recurrent Neural Network.

    Processes a sequence of inputs through an RNN cell.

    Args:
        input_size: Number of features in each input step.
        hidden_size: Number of features in the hidden state.
        num_layers: Number of stacked RNN layers.
        batch_first: If True, input shape is (batch, seq, features).
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, batch_first: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.cells = [RNNCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)]

    def forward(self, x: Any, h: Optional[Any] = None, **kwargs: Any) -> Any:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                if self.batch_first:
                    batch_size, seq_len, _ = x.shape
                else:
                    seq_len, batch_size, _ = x.shape
                    x = x.transpose(0, 1)

                if h is None:
                    h = [torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device) for _ in range(self.num_layers)]

                outputs = []
                for t in range(seq_len):
                    inp = x[:, t, :]
                    for layer_idx, cell in enumerate(self.cells):
                        h[layer_idx] = cell(inp, h[layer_idx])
                        inp = h[layer_idx]
                    outputs.append(inp)

                output = torch.stack(outputs, dim=1)
                return output, h[-1]
        except ImportError:
            pass

        return x, None


class LSTMCell(Layer):
    """LSTM cell implementing forget, input, output gates.

    Args:
        input_size: Number of expected input features.
        hidden_size: Number of features in the hidden state.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        gate_size = 4 * hidden_size
        limit = math.sqrt(6.0 / (input_size + hidden_size))
        self.w_ih = [[random.uniform(-limit, limit) for _ in range(gate_size)] for _ in range(input_size)]
        self.w_hh = [[random.uniform(-limit, limit) for _ in range(gate_size)] for _ in range(hidden_size)]
        self.bias = [0.0] * gate_size

    def forward(self, x: Any, state: Optional[Tuple[Any, Any]] = None, **kwargs: Any) -> Tuple[Any, Any]:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                w_ih = torch.tensor(self.w_ih, dtype=x.dtype, device=x.device)
                w_hh = torch.tensor(self.w_hh, dtype=x.dtype, device=x.device)
                b = torch.tensor(self.bias, dtype=x.dtype, device=x.device)
                batch_size = x.size(0) if x.dim() > 1 else 1
                if state is None:
                    h = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
                    c = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
                else:
                    h, c = state

                gates = x @ w_ih + h @ w_hh + b
                i_gate = torch.sigmoid(gates[:, : self.hidden_size])
                f_gate = torch.sigmoid(gates[:, self.hidden_size : 2 * self.hidden_size])
                g_gate = torch.tanh(gates[:, 2 * self.hidden_size : 3 * self.hidden_size])
                o_gate = torch.sigmoid(gates[:, 3 * self.hidden_size :])

                c_new = f_gate * c + i_gate * g_gate
                h_new = o_gate * torch.tanh(c_new)
                return h_new, c_new
        except ImportError:
            pass

        return ([0.0] * self.hidden_size, [0.0] * self.hidden_size)


class LSTM(Layer):
    """Long Short-Term Memory network.

    Args:
        input_size: Number of expected input features.
        hidden_size: Number of features in the hidden state.
        num_layers: Number of stacked LSTM layers.
        batch_first: If True, input shape is (batch, seq, features).
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, batch_first: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.cells = [LSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)]

    def forward(self, x: Any, state: Optional[Tuple[Any, Any]] = None, **kwargs: Any) -> Any:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                if self.batch_first:
                    batch_size, seq_len, _ = x.shape
                else:
                    seq_len, batch_size, _ = x.shape
                    x = x.transpose(0, 1)

                if state is None:
                    h_list = [torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device) for _ in range(self.num_layers)]
                    c_list = [torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device) for _ in range(self.num_layers)]
                else:
                    h_list, c_list = [s.clone() for s in state[0]], [s.clone() for s in state[1]]

                outputs = []
                for t in range(seq_len):
                    inp = x[:, t, :]
                    for layer_idx, cell in enumerate(self.cells):
                        h_list[layer_idx], c_list[layer_idx] = cell(inp, (h_list[layer_idx], c_list[layer_idx]))
                        inp = h_list[layer_idx]
                    outputs.append(inp)

                output = torch.stack(outputs, dim=1)
                return output, (h_list[-1], c_list[-1])
        except ImportError:
            pass

        return x, None


class GRUCell(Layer):
    """Gated Recurrent Unit cell.

    Args:
        input_size: Number of expected input features.
        hidden_size: Number of features in the hidden state.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        gate_size = 3 * hidden_size
        limit = math.sqrt(6.0 / (input_size + hidden_size))
        self.w_ih = [[random.uniform(-limit, limit) for _ in range(gate_size)] for _ in range(input_size)]
        self.w_hh = [[random.uniform(-limit, limit) for _ in range(gate_size)] for _ in range(hidden_size)]
        self.bias = [0.0] * gate_size

    def forward(self, x: Any, h: Optional[Any] = None, **kwargs: Any) -> Any:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                w_ih = torch.tensor(self.w_ih, dtype=x.dtype, device=x.device)
                w_hh = torch.tensor(self.w_hh, dtype=x.dtype, device=x.device)
                b = torch.tensor(self.bias, dtype=x.dtype, device=x.device)
                batch_size = x.size(0) if x.dim() > 1 else 1
                if h is None:
                    h = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)

                x_gates = x @ w_ih + b
                h_gates = h @ w_hh

                r = torch.sigmoid(x_gates[:, : self.hidden_size] + h_gates[:, : self.hidden_size])
                z = torch.sigmoid(x_gates[:, self.hidden_size : 2 * self.hidden_size] + h_gates[:, self.hidden_size : 2 * self.hidden_size])
                n = torch.tanh(x_gates[:, 2 * self.hidden_size :] + r * h_gates[:, 2 * self.hidden_size :])

                h_new = (1 - z) * n + z * h
                return h_new
        except ImportError:
            pass

        return [0.0] * self.hidden_size


class GRU(Layer):
    """Gated Recurrent Unit network.

    Args:
        input_size: Number of expected input features.
        hidden_size: Number of features in the hidden state.
        num_layers: Number of stacked GRU layers.
        batch_first: If True, input shape is (batch, seq, features).
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, batch_first: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.cells = [GRUCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)]

    def forward(self, x: Any, h: Optional[Any] = None, **kwargs: Any) -> Any:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                if self.batch_first:
                    batch_size, seq_len, _ = x.shape
                else:
                    seq_len, batch_size, _ = x.shape
                    x = x.transpose(0, 1)

                if h is None:
                    h_list = [torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device) for _ in range(self.num_layers)]
                else:
                    h_list = [h[i] for i in range(self.num_layers)]

                outputs = []
                for t in range(seq_len):
                    inp = x[:, t, :]
                    for layer_idx, cell in enumerate(self.cells):
                        h_list[layer_idx] = cell(inp, h_list[layer_idx])
                        inp = h_list[layer_idx]
                    outputs.append(inp)

                output = torch.stack(outputs, dim=1)
                return output, h_list[-1]
        except ImportError:
            pass

        return x, None


class Dropout(Layer):
    """Dropout layer."""

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Any, **kwargs: Any) -> Any:
        if not self.training or self.p == 0.0:
            return x

        try:
            import torch

            if isinstance(x, torch.Tensor):
                return torch.nn.functional.dropout(x, p=self.p, training=self.training)
        except ImportError:
            pass

        return x


class Flatten(Layer):
    """Flatten layer that reshapes input to 2D (batch_size, -1)."""

    def forward(self, x: Any, **kwargs: Any) -> Any:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                return x.view(x.size(0), -1)
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(x, np.ndarray):
                return x.reshape(x.shape[0], -1)
        except ImportError:
            pass

        return x


class Residual(Layer):
    """Residual connection layer wrapping a sub-module."""

    def __init__(self, fn: Layer) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Any, **kwargs: Any) -> Any:
        return x + self.fn(x, **kwargs)


class Sequential(Layer):
    """Sequential container executing layers in order."""

    def __init__(self, *layers: Layer) -> None:
        super().__init__()
        self.layers: List[Layer] = list(layers)

    def forward(self, x: Any, **kwargs: Any) -> Any:
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x

    def append(self, layer: Layer) -> None:
        """Add a layer to the end of the sequence."""
        self.layers.append(layer)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, idx: int) -> Layer:
        return self.layers[idx]
