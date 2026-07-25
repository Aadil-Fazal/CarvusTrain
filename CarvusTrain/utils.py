"""Utility functions for device detection, reproducibility, system monitoring, and formatting."""

import os
import platform
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union


def detect_device(requested: str = "auto") -> str:
    """Detect available compute device (CUDA GPU, Apple Silicon MPS, CPU).

    Args:
        requested: Requested device identifier ('auto', 'cpu', 'cuda', 'gpu', 'mps').

    Returns:
        Canonical string representing detected device ('cpu', 'cuda', 'mps').
    """
    req = str(requested).lower().strip()
    if req in ("cpu",):
        return "cpu"

    # Check PyTorch availability for GPU/MPS if present
    try:
        import torch

        if req in ("cuda", "gpu"):
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        if req == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        if req == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
    except ImportError:
        pass

    return "cpu"


def set_seed(seed: int = 42) -> None:
    """Set global random seed across Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def format_bytes(num_bytes: Union[int, float]) -> str:
    """Convert raw byte count into human-readable string.

    Args:
        num_bytes: Size in bytes.

    Returns:
        Human-readable formatted string (e.g. '1.24 MB').
    """
    if num_bytes < 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def format_time(seconds: float) -> str:
    """Format elapsed seconds into readable time string.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        Formatted time string (e.g. '01:23:45' or '12.34s').
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def get_system_info() -> Dict[str, Any]:
    """Retrieve detailed system hardware and software information.

    Returns:
        Dictionary containing platform, python version, CPU count, RAM, GPU status.
    """
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "processor": platform.processor() or "Unknown CPU",
        "cpu_count": os.cpu_count() or 1,
        "cuda_available": False,
        "cuda_device_name": None,
        "mps_available": False,
        "pytorch_installed": False,
        "numpy_installed": False,
    }

    try:
        import numpy as np

        info["numpy_installed"] = True
        info["numpy_version"] = np.__version__
    except ImportError:
        pass

    try:
        import torch

        info["pytorch_installed"] = True
        info["pytorch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_device_count"] = torch.cuda.device_count()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["mps_available"] = True
    except ImportError:
        pass

    return info


def detect_file_encoding(file_path: str) -> str:
    """Detect text encoding of a file by probing BOMs and common standard encodings.

    Args:
        file_path: Path to the target file.

    Returns:
        Detected encoding string (e.g. 'utf-8', 'utf-16', 'ascii').
    """
    if not os.path.isfile(file_path):
        return "utf-8"

    with open(file_path, "rb") as f:
        raw = f.read(4)

    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return "utf-8"
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"

    for encoding in ["utf-8", "ascii", "latin-1"]:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                f.read(1024)
            return encoding
        except (UnicodeDecodeError, Exception):
            continue

    return "utf-8"
