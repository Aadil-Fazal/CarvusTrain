"""Multi-format data and configuration parser for CarvusTrain."""

import csv
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple, Union

from .constants import (
    CARVUSTRAIN_SECTIONS,
    SECTION_HEADER_CARVUSTRAIN,
    SECTION_HEADER_EXAMPLES,
    SECTION_HEADER_KNOWLEDGE,
    SECTION_HEADER_MODEL,
    SECTION_HEADER_TRAINING,
)
from .exceptions import ParserError
from .utils import detect_file_encoding


class CarvusTrainFileContent:
    """Structured container for parsed .ct or custom CarvusTrain text files."""

    def __init__(self) -> None:
        self.model_config: Dict[str, str] = {}
        self.training_config: Dict[str, str] = {}
        self.carvustrain_config: Dict[str, str] = {}
        self.knowledge: List[str] = []
        self.examples: List[str] = []
        self.raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "carvustrain": self.carvustrain_config,
            "model": self.model_config,
            "training": self.training_config,
            "knowledge": self.knowledge,
            "examples": self.examples,
        }


class CarvusTrainParser:
    """Parser for native custom CarvusTrain sectioned files ([CarvusTrain], [Model], [Training], [Knowledge])."""

    @staticmethod
    def parse_text(text: str) -> CarvusTrainFileContent:
        """Parse raw sectioned CarvusTrain text.

        Args:
            text: Raw string content containing section headers like [Model], [Training], [Knowledge], [Examples].

        Returns:
            CarvusTrainFileContent object.
        """
        content = CarvusTrainFileContent()
        content.raw_text = text

        current_section = None
        knowledge_lines = []
        examples_lines = []

        lines = text.splitlines()
        for line in lines:
            stripped = line.strip()
            if not stripped and current_section not in (SECTION_HEADER_KNOWLEDGE, SECTION_HEADER_EXAMPLES):
                continue

            # Check section header
            if stripped.startswith("[") and stripped.endswith("]"):
                header = stripped
                # Normalize: match lowercased header against all known sections
                h_lower = header.lower()
                matched = None
                for s in CARVUSTRAIN_SECTIONS:
                    if h_lower == s.lower():
                        matched = s
                        break
                if matched:
                    current_section = matched
                    continue

            if current_section == SECTION_HEADER_KNOWLEDGE:
                knowledge_lines.append(line)
            elif current_section == SECTION_HEADER_EXAMPLES:
                examples_lines.append(line)
            elif current_section in (SECTION_HEADER_MODEL, "[Model]", "[model]"):
                if "=" in stripped:
                    k, v = stripped.split("=", 1)
                    content.model_config[k.strip()] = v.strip()
            elif current_section in (SECTION_HEADER_TRAINING, "[Training]", "[training]"):
                if "=" in stripped:
                    k, v = stripped.split("=", 1)
                    key_clean = k.strip()
                    val_clean = v.strip()
                    content.training_config[key_clean] = val_clean
                    if key_clean.lower() == "order":
                        content.training_config["shuffle"] = "true" if val_clean.lower() in ("random", "shuffle") else "false"
            elif current_section in (SECTION_HEADER_CARVUSTRAIN, "[CarvusTrain]", "[carvustrain]"):
                if "=" in stripped:
                    k, v = stripped.split("=", 1)
                    content.carvustrain_config[k.strip()] = v.strip()
            else:
                # If no section matched yet, treat line as unsectioned knowledge text
                knowledge_lines.append(line)

        content.knowledge = [k for k in "\n".join(knowledge_lines).split("\n\n") if k.strip()]
        content.examples = [e for e in "\n".join(examples_lines).split("\n\n") if e.strip()]
        return content

    @classmethod
    def parse_file(cls, file_path: str, encoding: Optional[str] = None) -> CarvusTrainFileContent:
        """Parse a CarvusTrain file on disk."""
        if not os.path.exists(file_path):
            raise ParserError(f"CarvusTrain file not found: {file_path}")

        enc = encoding or detect_file_encoding(file_path)
        try:
            with open(file_path, "r", encoding=enc) as f:
                text = f.read()
            return cls.parse_text(text)
        except Exception as e:
            raise ParserError(f"Failed to parse CarvusTrain file '{file_path}'", details=str(e))


class DataParser:
    """Unified multi-format file and directory dataset parser."""

    @classmethod
    def parse(cls, target_path: str, encoding: Optional[str] = None, recursive: bool = True) -> List[Dict[str, Any]]:
        """Parse target path (file or directory) into standardized list of sample records.

        Args:
            target_path: Path to single file or directory.
            encoding: Text encoding override.
            recursive: Whether to scan directory recursively.

        Returns:
            List of dictionary sample records containing 'text' or feature fields.
        """
        if not os.path.exists(target_path):
            raise ParserError(f"Dataset path does not exist: {target_path}")

        if os.path.isdir(target_path):
            return cls.parse_directory(target_path, encoding=encoding, recursive=recursive)

        return cls.parse_file(target_path, encoding=encoding)

    @classmethod
    def parse_file(cls, file_path: str, encoding: Optional[str] = None) -> List[Dict[str, Any]]:
        """Parse a single data file into sample records based on file extension."""
        enc = encoding or detect_file_encoding(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext in (".txt", ".text"):
                return cls._parse_txt(file_path, enc)
            elif ext == ".csv":
                return cls._parse_csv(file_path, enc)
            elif ext == ".json":
                return cls._parse_json(file_path, enc)
            elif ext == ".jsonl":
                return cls._parse_jsonl(file_path, enc)
            elif ext == ".xml":
                return cls._parse_xml(file_path, enc)
            elif ext in (".yaml", ".yml"):
                return cls._parse_yaml(file_path, enc)
            elif ext in (".md", ".markdown"):
                return cls._parse_markdown(file_path, enc)
            elif ext == ".ct":
                ct_content = CarvusTrainParser.parse_file(file_path, encoding=enc)
                records = [{"text": k, "source": file_path} for k in ct_content.knowledge]
                # Also include examples as training records
                for ex in ct_content.examples:
                    records.append({"text": ex, "source": file_path, "type": "example"})
                return records
            elif ext == ".cl":
                cl_content = CarvusTrainParser.parse_file(file_path, encoding=enc)
                records = [{"text": k, "source": file_path} for k in cl_content.knowledge]
                # Include examples — these are the primary learning material in .cl files
                for ex in cl_content.examples:
                    records.append({"text": ex, "source": file_path, "type": "example"})
                return records if records else [{"text": cl_content.raw_text, "source": file_path}]
            else:
                # Default text fallback
                return cls._parse_txt(file_path, enc)
        except Exception as e:
            if isinstance(e, ParserError):
                raise e
            raise ParserError(f"Error parsing file '{file_path}' (format: {ext})", details=str(e))

    @classmethod
    def parse_directory(cls, dir_path: str, encoding: Optional[str] = None, recursive: bool = True) -> List[Dict[str, Any]]:
        """Scan directory and parse all supported dataset files."""
        records: List[Dict[str, Any]] = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                full_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                if ext in (".txt", ".csv", ".json", ".jsonl", ".xml", ".yaml", ".yml", ".md", ".ct", ".cl"):
                    try:
                        file_records = cls.parse_file(full_path, encoding=encoding)
                        records.extend(file_records)
                    except Exception:
                        continue
            if not recursive:
                break
        return records

    @staticmethod
    def _parse_txt(file_path: str, encoding: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            text = f.read()

        # Split into paragraphs or non-empty lines
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [line.strip() for line in text.splitlines() if line.strip()]

        return [{"text": p, "source": file_path} for p in paragraphs]

    @staticmethod
    def _parse_csv(file_path: str, encoding: str) -> List[Dict[str, Any]]:
        records = []
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Combine row values into text if no explicit text column
                text = row.get("text") or row.get("content") or " ".join(str(v) for v in row.values())
                record = dict(row)
                record["text"] = text
                record["source"] = file_path
                records.append(record)
        return records

    @staticmethod
    def _parse_json(file_path: str, encoding: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            data = json.load(f)

        if isinstance(data, list):
            records = []
            for item in data:
                if isinstance(item, dict):
                    rec = dict(item)
                    rec["text"] = item.get("text") or item.get("prompt") or str(item)
                    records.append(rec)
                else:
                    records.append({"text": str(item), "source": file_path})
            return records
        elif isinstance(data, dict):
            text = data.get("text") or data.get("content") or json.dumps(data)
            return [{"text": text, "data": data, "source": file_path}]
        return [{"text": str(data), "source": file_path}]

    @staticmethod
    def _parse_jsonl(file_path: str, encoding: str) -> List[Dict[str, Any]]:
        records = []
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                    if isinstance(obj, dict):
                        rec = dict(obj)
                        rec["text"] = obj.get("text") or obj.get("prompt") or str(obj)
                        records.append(rec)
                    else:
                        records.append({"text": str(obj), "source": file_path})
                except json.JSONDecodeError:
                    continue
        return records

    @staticmethod
    def _parse_xml(file_path: str, encoding: str) -> List[Dict[str, Any]]:
        tree = ET.parse(file_path)
        root = tree.getroot()
        records = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                records.append({
                    "tag": elem.tag,
                    "text": elem.text.strip(),
                    "attrib": elem.attrib,
                    "source": file_path,
                })
        return records

    @staticmethod
    def _parse_yaml(file_path: str, encoding: str) -> List[Dict[str, Any]]:
        try:
            import yaml

            with open(file_path, "r", encoding=encoding, errors="replace") as f:
                data = yaml.safe_load(f)
            if isinstance(data, list):
                return [{"text": str(item), "data": item, "source": file_path} for item in data]
            return [{"text": str(data), "data": data, "source": file_path}]
        except ImportError:
            # Fallback simple line reader
            return DataParser._parse_txt(file_path, encoding)

    @staticmethod
    def _parse_markdown(file_path: str, encoding: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            content = f.read()

        # Remove markdown header syntax for clean text
        clean_text = re.sub(r"#+\s*", "", content)
        clean_text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", clean_text)
        paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip()]

        return [{"text": p, "raw_markdown": content, "source": file_path} for p in paragraphs]
