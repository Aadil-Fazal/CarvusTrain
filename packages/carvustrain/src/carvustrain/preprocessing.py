"""Text preprocessing, cleaning, normalization, and sequence manipulation utilities."""

import re
import unicodedata
from typing import List, Optional, Tuple, Union


class TextPreprocessor:
    """Configurable text preprocessor for cleaning, normalizing, and chunking text."""

    def __init__(
        self,
        lowercase: bool = False,
        strip_accents: bool = True,
        remove_html_tags: bool = True,
        normalize_whitespace: bool = True,
        replace_numbers: bool = False,
    ) -> None:
        self.lowercase = lowercase
        self.strip_accents = strip_accents
        self.remove_html_tags = remove_html_tags
        self.normalize_whitespace = normalize_whitespace
        self.replace_numbers = replace_numbers

    def clean(self, text: str) -> str:
        """Clean and normalize a single text string.

        Args:
            text: Raw input string.

        Returns:
            Cleaned and normalized text string.
        """
        if not text:
            return ""

        # Remove HTML tags if requested
        if self.remove_html_tags:
            text = re.sub(r"<[^>]+>", " ", text)

        # Unicode normalization & strip accents
        if self.strip_accents:
            text = unicodedata.normalize("NFD", text)
            text = "".join(c for c in text if unicodedata.category(c) != "Mn")
            text = unicodedata.normalize("NFC", text)

        # Replace numbers
        if self.replace_numbers:
            text = re.sub(r"\d+", "<NUM>", text)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Whitespace normalization
        if self.normalize_whitespace:
            text = re.sub(r"\s+", " ", text).strip()

        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a batch of text strings."""
        return [self.clean(t) for t in texts]


def pad_sequence(
    sequence: List[int],
    max_length: int,
    pad_value: int = 0,
    truncate_side: str = "right",
    pad_side: str = "right",
) -> List[int]:
    """Pad or truncate a list of integer token IDs to a fixed length.

    Args:
        sequence: List of integer token IDs.
        max_length: Target sequence length.
        pad_value: Token ID used for padding.
        truncate_side: 'left' or 'right' truncation.
        pad_side: 'left' or 'right' padding.

    Returns:
        List of padded or truncated token IDs of length max_length.
    """
    if len(sequence) > max_length:
        if truncate_side == "left":
            return sequence[-max_length:]
        return sequence[:max_length]

    pad_amount = max_length - len(sequence)
    if pad_amount == 0:
        return list(sequence)

    padding = [pad_value] * pad_amount
    if pad_side == "left":
        return padding + sequence
    return sequence + padding


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[str]:
    """Split text into overlapping windows of fixed character/word length.

    Args:
        text: Input text string.
        chunk_size: Maximum length of each chunk.
        overlap: Overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break

    return chunks
