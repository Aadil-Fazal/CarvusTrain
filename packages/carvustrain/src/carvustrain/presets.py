"""Multi-billion parameter architecture presets for CarvusTrain.

These presets configure model dimensions for scales from tiny research
models to frontier multi-hundred-billion parameter layouts. They do not
download weights — they size the training/inference configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ParameterPreset:
    """Architecture sizing for a named parameter scale."""

    name: str
    approx_params: int
    vocab_size: int
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    max_seq_len: int
    dropout: float = 0.1

    @property
    def params_billions(self) -> float:
        return self.approx_params / 1_000_000_000

    def label(self) -> str:
        if self.approx_params >= 1_000_000_000:
            return f"{self.params_billions:.1f}B".replace(".0B", "B")
        if self.approx_params >= 1_000_000:
            return f"{self.approx_params / 1_000_000:.0f}M"
        return f"{self.approx_params}"


# Named scales for multi-billion-parameter library usage
PARAMETER_PRESETS: Dict[str, ParameterPreset] = {
    "tiny": ParameterPreset("tiny", 15_000_000, 8_000, 256, 512, 4, 4, 512, 0.1),
    "base": ParameterPreset("base", 125_000_000, 32_000, 768, 3072, 12, 12, 2048, 0.1),
    "350M": ParameterPreset("350M", 350_000_000, 32_000, 1024, 4096, 24, 16, 2048, 0.1),
    "1B": ParameterPreset("1B", 1_000_000_000, 32_000, 2048, 5504, 24, 16, 4096, 0.05),
    "3B": ParameterPreset("3B", 3_000_000_000, 32_000, 3072, 8192, 32, 24, 4096, 0.05),
    "7B": ParameterPreset("7B", 7_000_000_000, 32_000, 4096, 11008, 32, 32, 8192, 0.05),
    "13B": ParameterPreset("13B", 13_000_000_000, 32_000, 5120, 13824, 40, 40, 8192, 0.05),
    "70B": ParameterPreset("70B", 70_000_000_000, 32_000, 8192, 28672, 80, 64, 8192, 0.05),
    "405B": ParameterPreset("405B", 405_000_000_000, 128_000, 16384, 53248, 126, 128, 16384, 0.0),
}


def resolve_preset(parameters: str) -> ParameterPreset:
    """Resolve a parameters string (e.g. '7B', '7b', 'base') to a preset."""
    key = str(parameters).strip()
    if key in PARAMETER_PRESETS:
        return PARAMETER_PRESETS[key]
    lower = key.lower()
    for name, preset in PARAMETER_PRESETS.items():
        if name.lower() == lower:
            return preset
    # aliases
    aliases = {
        "small": "base",
        "medium": "1B",
        "large": "7B",
        "xl": "13B",
        "xxl": "70B",
        "frontier": "405B",
    }
    if lower in aliases:
        return PARAMETER_PRESETS[aliases[lower]]
    return PARAMETER_PRESETS["base"]


def list_presets() -> Dict[str, str]:
    """Return preset name → human label map."""
    return {name: p.label() for name, p in PARAMETER_PRESETS.items()}


def estimate_memory_bytes(preset: ParameterPreset, dtype_bytes: int = 2) -> Dict[str, float]:
    """Rough memory estimates for weights + activations (GB)."""
    weight_bytes = preset.approx_params * dtype_bytes
    # activations rough: batch=1, seq, hidden
    activation_bytes = preset.max_seq_len * preset.hidden_dim * dtype_bytes * preset.num_layers
    total = weight_bytes + activation_bytes
    return {
        "weights_gb": weight_bytes / (1024**3),
        "activations_gb": activation_bytes / (1024**3),
        "total_gb": total / (1024**3),
        "dtype_bytes": dtype_bytes,
    }
