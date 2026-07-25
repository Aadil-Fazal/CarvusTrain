"""CarvusTrain - High Performance AI Model Training and Inference Framework."""

from typing import Any, Dict, List, Optional, Union

from . import activation, callbacks, layers, losses, metrics
# RAG module — optional, install manually when carvustrain/rag.py exists
try:
    from . import rag
except ImportError:
    pass
from .configuration import (
    CarvusConfig,
    DatasetConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
    load_config_file,
)
from .dataset import DataLoader, Dataset, TabularDataset, TextDataset
from .evaluation import Benchmarker, Evaluator
from .exceptions import (
    CarvusTrainError,
    ConfigurationError,
    DatasetNotFoundError,
    ExportError,
    InferenceError,
    ModelError,
    ParserError,
    TokenizerError,
    TrainingError,
)
from .exporter import ModelExporter
from .logger import logger
from .model import (
    AgentModel,
    ChatModel,
    CustomModel,
    LanguageModel,
    Model,
    TextModel,
)
from .optimizer import SGD, AdaGrad, Adam, AdamW, Optimizer, RMSprop
from .parser import CarvusTrainParser, DataParser
from .scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LinearWarmupLR,
    ReduceLROnPlateau,
    StepLR,
)
from .tokenizer import (
    BPETokenizer,
    CharTokenizer,
    CustomTokenizer,
    SentencePieceTokenizer,
    SentenceTokenizer,
    Tokenizer,
    Vocabulary,
    WordPieceTokenizer,
    WordTokenizer,
)
from .trainer import Trainer
from .version import __version__

# Global default model instance for functional top-level API calls
_default_model: Optional[Model] = None


def _get_default_model() -> Model:
    global _default_model
    if _default_model is None:
        _default_model = Model(name="CarvusGlobal")
    return _default_model


def train(data: Optional[Union[str, Dataset, List[str]]] = None, **kwargs: Any) -> Dict[str, Any]:
    """Train global default model instance."""
    return _get_default_model().train(data=data, **kwargs)


def ask(question: str) -> str:
    """Ask question to global default model instance."""
    return _get_default_model().ask(question)


def predict(input_text: str) -> str:
    """Generate prediction using global default model instance."""
    return _get_default_model().predict(input_text)


def generate(prompt: str, **kwargs: Any) -> str:
    """Generate text using global default model instance."""
    return _get_default_model().generate(prompt, **kwargs)


def chat(user_message: str) -> str:
    """Chat with global default model instance."""
    return _get_default_model().chat(user_message)


def learn(text: Union[str, List[str]]) -> None:
    """Ingest knowledge into global default model instance."""
    _get_default_model().learn(text)


def finetune(data: Union[str, Dataset, List[str]], **kwargs: Any) -> Dict[str, Any]:
    """Fine-tune global default model instance."""
    return _get_default_model().finetune(data=data, **kwargs)


def evaluate(dataset: Optional[Dataset] = None) -> Dict[str, float]:
    """Evaluate global default model instance."""
    return _get_default_model().evaluate(dataset=dataset)


def export(output_path: str, format: Optional[str] = None) -> str:
    """Export global default model instance."""
    return _get_default_model().export(output_path=output_path, format=format)


def save(filepath: Optional[str] = None) -> str:
    """Save global default model instance."""
    return _get_default_model().save(filepath=filepath)


def load(filepath: str) -> Model:
    """Load model state from filepath."""
    return _get_default_model().load(filepath=filepath)


def summary() -> None:
    """Display summary of global default model instance."""
    _get_default_model().summary()


def statistics() -> Dict[str, Any]:
    """Get statistics of global default model instance."""
    return _get_default_model().statistics()


def memory() -> Dict[str, Any]:
    """Inspect memory of global default model instance."""
    return _get_default_model().memory()


__all__ = [
    # Main Classes
    "Model",
    "ChatModel",
    "TextModel",
    "LanguageModel",
    "CustomModel",
    # Data & Tokenization
    "Dataset",
    "TextDataset",
    "TabularDataset",
    "DataLoader",
    "Tokenizer",
    "Vocabulary",
    "WordTokenizer",
    "CharTokenizer",
    "SentenceTokenizer",
    "BPETokenizer",
    "WordPieceTokenizer",
    "SentencePieceTokenizer",
    "CustomTokenizer",
    "DataParser",
    "CarvusTrainParser",
    # Engine & Optimization
    "Trainer",
    "Optimizer",
    "Adam",
    "AdamW",
    "SGD",
    "RMSprop",
    "AdaGrad",
    "StepLR",
    "CosineAnnealingLR",
    "LinearWarmupLR",
    "ExponentialLR",
    "ReduceLROnPlateau",
    "ModelExporter",
    "Evaluator",
    "Benchmarker",
    # Configuration & Logging
    "CarvusConfig",
    "ModelConfig",
    "TrainingConfig",
    "DatasetConfig",
    "InferenceConfig",
    "load_config_file",
    "logger",
    "__version__",
    # Exceptions
    "CarvusTrainError",
    "DatasetNotFoundError",
    "TrainingError",
    "ModelError",
    "ParserError",
    "TokenizerError",
    "ExportError",
    "InferenceError",
    "ConfigurationError",
    # Modules
    "activation",
    "callbacks",
    "layers",
    "losses",
    "metrics",
    # Functional APIs
    "train",
    "ask",
    "predict",
    "generate",
    "chat",
    "learn",
    "finetune",
    "evaluate",
    "export",
    "save",
    "load",
    "summary",
    "statistics",
    "memory",
]
