"""Exporter engine for serializing CarvusTrain models to .ct, .bin, .onnx, .json, and .gguf formats."""

import json
import os
import struct
import zipfile
from typing import Any, Dict, Optional

from .constants import SUPPORTED_EXPORT_FORMATS
from .exceptions import ExportError
from .logger import logger


class ModelExporter:
    """Exports CarvusTrain models to various binary, neural exchange, and open formats."""

    @classmethod
    def export(cls, model: Any, output_path: str, format: Optional[str] = None) -> str:
        """Export model to target output file path.

        Args:
            model: Model instance to export.
            output_path: Target output file path.
            format: Target format ('ct', 'bin', 'onnx', 'json', 'gguf'). Inferred from extension if None.

        Returns:
            Absolute path to exported model file.

        Raises:
            ExportError: If format is unsupported or export operation fails.
        """
        if not output_path:
            raise ExportError("Output path must be specified.")

        fmt = format.lower().strip() if format else os.path.splitext(output_path)[1].lstrip(".").lower()
        if not fmt:
            fmt = "ct"

        if fmt not in SUPPORTED_EXPORT_FORMATS:
            raise ExportError(f"Unsupported export format '{fmt}'. Supported formats: {SUPPORTED_EXPORT_FORMATS}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        logger.info(f"Exporting model '{getattr(model, 'name', 'Carvus')}' to format '{fmt}' at '{output_path}'...")

        try:
            if fmt == "ct":
                cls._export_ct(model, output_path)
            elif fmt == "json":
                cls._export_json(model, output_path)
            elif fmt == "bin":
                cls._export_bin(model, output_path)
            elif fmt == "onnx":
                cls._export_onnx(model, output_path)
            elif fmt == "gguf":
                cls._export_gguf(model, output_path)
            else:
                cls._export_ct(model, output_path)

            logger.success(f"Successfully exported model to '{output_path}'")
            return os.path.abspath(output_path)
        except Exception as e:
            if isinstance(e, ExportError):
                raise e
            raise ExportError(f"Failed to export model to '{output_path}'", details=str(e))

    @staticmethod
    def _export_ct(model: Any, output_path: str) -> None:
        """Export as native .ct CarvusTrain zip archive."""
        model_data = model.to_dict() if hasattr(model, "to_dict") else {"name": getattr(model, "name", "Carvus")}
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("model_manifest.json", json.dumps(model_data, indent=2))
            zf.writestr("version.txt", "CarvusTrain v0.1.0")

    @staticmethod
    def _export_json(model: Any, output_path: str) -> None:
        """Export model manifest, parameters, and knowledge to JSON."""
        model_data = model.to_dict() if hasattr(model, "to_dict") else {"name": getattr(model, "name", "Carvus")}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=2)

    @staticmethod
    def _export_bin(model: Any, output_path: str) -> None:
        """Export as raw binary file."""
        model_data = model.to_dict() if hasattr(model, "to_dict") else {"name": getattr(model, "name", "Carvus")}
        raw_bytes = json.dumps(model_data).encode("utf-8")
        header = struct.pack("<4sI", b"CTBN", len(raw_bytes))
        with open(output_path, "wb") as f:
            f.write(header + raw_bytes)

    @staticmethod
    def _export_onnx(model: Any, output_path: str) -> None:
        """Export ONNX format metadata descriptor."""
        onnx_meta = {
            "ir_version": 8,
            "producer_name": "CarvusTrain",
            "producer_version": "0.1.0",
            "domain": "ai.carvustrain",
            "model_version": 1,
            "doc_string": f"ONNX Export for CarvusTrain Model '{getattr(model, 'name', 'Carvus')}'",
            "graph": {
                "name": getattr(model, "name", "Carvus"),
                "inputs": [{"name": "input_ids", "type": "INT64", "shape": ["batch_size", "seq_len"]}],
                "outputs": [{"name": "logits", "type": "FLOAT", "shape": ["batch_size", "seq_len", "vocab_size"]}],
            },
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(onnx_meta, f, indent=2)

    @staticmethod
    def _export_gguf(model: Any, output_path: str) -> None:
        """Export GGUF format container file."""
        # GGUF Magic Header: GGUF (0x46554747)
        magic = b"GGUF"
        version = struct.pack("<I", 3)  # GGUF v3
        tensor_count = struct.pack("<Q", 0)
        kv_count = struct.pack("<Q", 1)

        model_name = getattr(model, "name", "Carvus").encode("utf-8")
        name_len = struct.pack("<Q", len(model_name))

        with open(output_path, "wb") as f:
            f.write(magic)
            f.write(version)
            f.write(tensor_count)
            f.write(kv_count)
            f.write(struct.pack("<Q", 8))  # Key len
            f.write(b"general.name")
            f.write(struct.pack("<I", 8))  # Value string type
            f.write(name_len)
            f.write(model_name)
