"""Configuration dataclasses and loaders for CarvusTrain."""

from dataclasses import asdict, dataclass, field
import json
import os
from typing import Any, Dict, Optional, Union

from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EPOCHS,
    DEFAULT_GRADIENT_CLIPPING,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_NUM_HEADS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_VOCAB_SIZE,
    DEFAULT_WEIGHT_DECAY,
)
from .exceptions import ConfigurationError


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    name: str = "Carvus"
    vocab_size: int = DEFAULT_VOCAB_SIZE
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    num_layers: int = DEFAULT_NUM_LAYERS
    num_heads: int = DEFAULT_NUM_HEADS
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    dropout: float = DEFAULT_DROPOUT
    model_type: str = "transformer"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    method: str = "normal"
    duration: Union[int, str] = 10
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    optimizer: str = "adamw"
    device: str = "auto"
    mixed_precision: bool = False
    workers: int = 0
    shuffle: bool = True
    seed: int = 42
    checkpoint: bool = True
    checkpoint_dir: str = "checkpoints"
    resume: bool = False
    validation_split: float = 0.1
    early_stopping: Optional[int] = None
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    gradient_clipping: float = DEFAULT_GRADIENT_CLIPPING

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""

    data_path: Optional[str] = None
    format: str = "auto"
    encoding: str = "auto"
    recursive: bool = True
    split_ratio: float = 0.8
    max_samples: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceConfig:
    """Configuration for inference and text generation."""

    max_new_tokens: int = 100
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CarvusConfig:
    """Master configuration encapsulating all CarvusTrain settings."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "dataset": self.dataset.to_dict(),
            "inference": self.inference.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CarvusConfig":
        model_cfg = ModelConfig(**data.get("model", {}))
        train_cfg = TrainingConfig(**data.get("training", {}))
        dataset_cfg = DatasetConfig(**data.get("dataset", {}))
        infer_cfg = InferenceConfig(**data.get("inference", {}))
        return cls(
            model=model_cfg,
            training=train_cfg,
            dataset=dataset_cfg,
            inference=infer_cfg,
        )


def load_config_file(file_path: str) -> CarvusConfig:
    """Load configuration from a JSON, YAML, or TOML file.

    Args:
        file_path: Path to configuration file.

    Returns:
        CarvusConfig instance populated with loaded settings.

    Raises:
        ConfigurationError: If format is unsupported or file is invalid.
    """
    if not os.path.exists(file_path):
        raise ConfigurationError(f"Configuration file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return CarvusConfig.from_dict(data)

        elif ext in (".yaml", ".yml"):
            try:
                import yaml

                with open(file_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                return CarvusConfig.from_dict(data or {})
            except ImportError:
                raise ConfigurationError("PyYAML is required to parse YAML config files. Install with 'pip install pyyaml'.")

        elif ext == ".toml":
            try:
                import tomllib

                with open(file_path, "rb") as f:
                    data = tomllib.load(f)
                return CarvusConfig.from_dict(data)
            except ImportError:
                import tomli

                with open(file_path, "rb") as f:
                    data = tomli.load(f)
                return CarvusConfig.from_dict(data)

        else:
            raise ConfigurationError(f"Unsupported configuration file extension: {ext}")

    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise e
        raise ConfigurationError(f"Failed to parse configuration file '{file_path}'", details=str(e))
