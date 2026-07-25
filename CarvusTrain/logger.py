"""Logging utility for CarvusTrain with colorized console output and system telemetry."""

import logging
import os
import sys
import time
from typing import Any, Dict, Optional


class ColorFormatter(logging.Formatter):
    """Custom colorized log formatter for terminal output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m\033[37m",  # White on Red
        "SUCCESS": "\033[32;1m",  # Bright Green
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        prefix = f"{self.BOLD}[CarvusTrain]{self.RESET} {timestamp} [{color}{record.levelname}{self.RESET}]"
        return f"{prefix} {record.getMessage()}"


class CarvusLogger:
    """Logger for CarvusTrain framework providing console, file, and metric logging."""

    _instance: Optional["CarvusLogger"] = None

    def __init__(self, name: str = "CarvusTrain", level: int = logging.INFO, log_file: Optional[str] = None) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(ColorFormatter())
            self.logger.addHandler(console_handler)

        if log_file:
            self.add_file_handler(log_file)

    @classmethod
    def get_logger(cls, name: str = "CarvusTrain") -> "CarvusLogger":
        """Get or initialize singleton logger instance.

        Args:
            name: Logger identifier string.

        Returns:
            Configured CarvusLogger instance.
        """
        if cls._instance is None:
            cls._instance = CarvusLogger(name=name)
        return cls._instance

    def add_file_handler(self, file_path: str) -> None:
        """Add a file logging handler.

        Args:
            file_path: Path to log file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def set_level(self, level: int) -> None:
        """Set logging verbosity level."""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug message."""
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an informational message."""
        self.logger.info(msg, *args, **kwargs)

    def success(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a success message."""
        self.logger.info(f"SUCCESS: {msg}", *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message."""
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a critical message."""
        self.logger.critical(msg, *args, **kwargs)

    def log_epoch(self, epoch: int, total_epochs: int, loss: float, accuracy: Optional[float] = None, lr: Optional[float] = None, elapsed: Optional[float] = None) -> None:
        """Log training epoch summary.

        Args:
            epoch: Current epoch (1-indexed).
            total_epochs: Total number of epochs.
            loss: Current epoch loss value.
            accuracy: Current epoch accuracy value.
            lr: Current learning rate.
            elapsed: Elapsed time in seconds.
        """
        metrics_str = f"Loss: {loss:.4f}"
        if accuracy is not None:
            metrics_str += f" | Acc: {accuracy:.4f} ({accuracy * 100:.2f}%)"
        if lr is not None:
            metrics_str += f" | LR: {lr:.6f}"
        if elapsed is not None:
            metrics_str += f" | Time: {elapsed:.2f}s"

        epoch_str = "FOREVER" if total_epochs < 0 else f"{epoch}/{total_epochs}"
        self.info(f"Epoch [{epoch_str}] - {metrics_str}")


logger = CarvusLogger.get_logger()
