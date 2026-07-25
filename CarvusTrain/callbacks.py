"""Callback infrastructure for monitoring, early stopping, checkpointing, and logging during training."""

import csv
import os
import time
from typing import Any, Dict, List, Optional

from .logger import logger
from .utils import format_bytes, format_time


class Callback:
    """Base callback class for training hooks."""

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving."""

    def __init__(self, monitor: str = "val_loss", patience: int = 5, min_delta: float = 1e-4, mode: str = "min") -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_val = float("inf") if mode == "min" else float("-inf")
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        is_better = (current < self.best_val - self.min_delta) if self.mode == "min" else (current > self.best_val + self.min_delta)
        if is_better:
            self.best_val = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.should_stop = True
                logger.warning(f"Early stopping triggered at epoch {epoch} (best {self.monitor}: {self.best_val:.4f})")


class ModelCheckpoint(Callback):
    """Save model checkpoint file at regular intervals or when metric improves."""

    def __init__(self, filepath: str = "checkpoints/checkpoint_epoch_{epoch}.ct", save_best_only: bool = True, monitor: str = "val_loss") -> None:
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.best_val = float("inf")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        model = logs.get("model")
        current = logs.get(self.monitor, logs.get("loss"))

        if model is None:
            return

        if self.save_best_only and current is not None:
            if current < self.best_val:
                self.best_val = current
                save_path = self.filepath.format(epoch=epoch)
                model.save(save_path)
                logger.info(f"Checkpoint saved to '{save_path}' ({self.monitor}: {current:.4f})")
        else:
            save_path = self.filepath.format(epoch=epoch)
            model.save(save_path)
            logger.info(f"Checkpoint saved to '{save_path}'")


class ProgressBarCallback(Callback):
    """Console progress logging for training iterations."""

    def __init__(self) -> None:
        self.start_time: float = 0.0

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self.start_time = time.time()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        elapsed = time.time() - self.start_time
        total_epochs = logs.get("total_epochs", 0)
        loss = logs.get("loss", 0.0)
        acc = logs.get("accuracy")
        lr = logs.get("lr")
        logger.log_epoch(epoch, total_epochs, loss, accuracy=acc, lr=lr, elapsed=elapsed)


class CSVLoggerCallback(Callback):
    """Log training progress to a CSV spreadsheet file."""

    def __init__(self, filename: str = "training_log.csv") -> None:
        self.filename = filename
        self.file_opened = False

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        row = {"epoch": epoch, "loss": logs.get("loss"), "accuracy": logs.get("accuracy"), "lr": logs.get("lr")}

        os.makedirs(os.path.dirname(os.path.abspath(self.filename)), exist_ok=True)
        file_exists = os.path.exists(self.filename)

        with open(self.filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


class MemoryTrackerCallback(Callback):
    """Monitors CPU/GPU memory footprint during training loop."""

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0)
                reserved = torch.cuda.memory_reserved(0)
                logger.debug(f"[Memory] GPU Allocated: {format_bytes(allocated)} | Reserved: {format_bytes(reserved)}")
        except ImportError:
            pass
