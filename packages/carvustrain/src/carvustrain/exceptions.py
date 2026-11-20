"""Custom exception classes for CarvusTrain."""

from typing import Optional


class CarvusTrainError(Exception):
    """Base exception class for all errors in CarvusTrain."""

    def __init__(self, message: str, details: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class DatasetNotFoundError(CarvusTrainError):
    """Raised when a dataset or specified data file cannot be located."""

    pass


class TrainingError(CarvusTrainError):
    """Raised when an error occurs during model training or optimization."""

    pass


class ModelError(CarvusTrainError):
    """Raised when an error occurs in model architecture, weight loading, or state management."""

    pass


class ParserError(CarvusTrainError):
    """Raised when parsing dataset files or configuration files fails."""

    pass


class TokenizerError(CarvusTrainError):
    """Raised when text tokenization or vocabulary construction fails."""

    pass


class ExportError(CarvusTrainError):
    """Raised when model export to a specific format (.ct, .bin, .onnx, .json, .gguf) fails."""

    pass


class InferenceError(CarvusTrainError):
    """Raised when text generation, prediction, or question-answering inference fails."""

    pass


class ConfigurationError(CarvusTrainError):
    """Raised when invalid parameters or configuration formats are provided."""

    pass
