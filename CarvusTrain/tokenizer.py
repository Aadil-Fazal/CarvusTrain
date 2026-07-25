"""Tokenizer suite for CarvusTrain implementing Word, Char, Sentence, BPE, WordPiece, and SentencePiece tokenizers."""

import collections
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .constants import (
    BOS_TOKEN,
    EOS_TOKEN,
    MASK_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    UNK_TOKEN,
)
from .exceptions import TokenizerError


class Vocabulary:
    """Bi-directional vocabulary mapping tokens to unique integer IDs and vice versa."""

    def __init__(self, special_tokens: Optional[List[str]] = None) -> None:
        self.tok2id: Dict[str, int] = {}
        self.id2tok: Dict[int, str] = {}
        self.special_tokens = special_tokens or SPECIAL_TOKENS.copy()

        # Add special tokens first
        for tok in self.special_tokens:
            self.add_token(tok)

    def add_token(self, token: str) -> int:
        """Add a token to vocabulary if not present and return its ID."""
        if token not in self.tok2id:
            idx = len(self.tok2id)
            self.tok2id[token] = idx
            self.id2tok[idx] = token
            return idx
        return self.tok2id[token]

    def get_id(self, token: str) -> int:
        """Get ID for token, returning UNK_TOKEN ID if not found."""
        return self.tok2id.get(token, self.tok2id.get(UNK_TOKEN, 1))

    def get_token(self, idx: int) -> str:
        """Get token string for ID, returning UNK_TOKEN if not found."""
        return self.id2tok.get(idx, UNK_TOKEN)

    def __len__(self) -> int:
        return len(self.tok2id)

    def __contains__(self, token: str) -> bool:
        return token in self.tok2id

    def to_dict(self) -> Dict[str, Any]:
        """Serialize vocabulary to dictionary."""
        return {
            "tok2id": self.tok2id,
            "special_tokens": self.special_tokens,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Vocabulary":
        """Reconstruct vocabulary from dictionary."""
        vocab = cls(special_tokens=data.get("special_tokens"))
        vocab.tok2id = data.get("tok2id", {})
        vocab.id2tok = {int(v): k for k, v in vocab.tok2id.items()}
        return vocab


class BaseTokenizer:
    """Abstract base class for all tokenizers in CarvusTrain."""

    def __init__(self, vocab: Optional[Vocabulary] = None) -> None:
        self.vocab = vocab or Vocabulary()

    def tokenize(self, text: str) -> List[str]:
        """Convert raw text into list of string tokens."""
        raise NotImplementedError

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert text into token IDs."""
        tokens = self.tokenize(text)
        ids = [self.vocab.get_id(t) for t in tokens]
        if add_special_tokens:
            bos_id = self.vocab.get_id(BOS_TOKEN)
            eos_id = self.vocab.get_id(EOS_TOKEN)
            return [bos_id] + ids + [eos_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert list of token IDs back into string text."""
        tokens = []
        for idx in ids:
            tok = self.vocab.get_token(idx)
            if skip_special_tokens and tok in SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return self._reconstruct_text(tokens)

    def _reconstruct_text(self, tokens: List[str]) -> str:
        """Helper to join tokens into readable text."""
        return " ".join(tokens)

    def train_on_texts(self, texts: List[str], max_vocab_size: int = 10000) -> None:
        """Build vocabulary from a corpus of texts."""
        pass


class WordTokenizer(BaseTokenizer):
    """Simple word-level whitespace and punctuation tokenizer."""

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return tokens

    def train_on_texts(self, texts: List[str], max_vocab_size: int = 10000) -> None:
        counts: collections.Counter = collections.Counter()
        for text in texts:
            counts.update(self.tokenize(text))

        most_common = counts.most_common(max_vocab_size - len(self.vocab))
        for token, _ in most_common:
            self.vocab.add_token(token)


class CharTokenizer(BaseTokenizer):
    """Character-level tokenizer."""

    def tokenize(self, text: str) -> List[str]:
        return list(text) if text else []

    def _reconstruct_text(self, tokens: List[str]) -> str:
        return "".join(tokens)

    def train_on_texts(self, texts: List[str], max_vocab_size: int = 10000) -> None:
        chars: Set[str] = set()
        for text in texts:
            chars.update(text)
        for char in sorted(chars):
            if len(self.vocab) >= max_vocab_size:
                break
            self.vocab.add_token(char)


class SentenceTokenizer(BaseTokenizer):
    """Sentence-level tokenizer splitting on sentence boundaries."""

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def train_on_texts(self, texts: List[str], max_vocab_size: int = 10000) -> None:
        for text in texts:
            for s in self.tokenize(text):
                if len(self.vocab) >= max_vocab_size:
                    break
                self.vocab.add_token(s)


class BPETokenizer(BaseTokenizer):
    """Byte Pair Encoding (BPE) subword tokenizer."""

    def __init__(self, vocab: Optional[Vocabulary] = None, merges: Optional[List[Tuple[str, str]]] = None) -> None:
        super().__init__(vocab)
        self.merges: List[Tuple[str, str]] = merges or []

    def train_on_texts(self, texts: List[str], max_vocab_size: int = 10000, num_merges: int = 1000) -> None:
        word_counts: collections.Counter = collections.Counter()
        for text in texts:
            words = text.split()
            for w in words:
                chars = tuple(list(w) + ["</w>"])
                word_counts[chars] += 1

        # Train BPE merges
        for _ in range(num_merges):
            pairs: collections.Counter = collections.Counter()
            for word, freq in word_counts.items():
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i + 1])] += freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)  # type: ignore
            self.merges.append(best_pair)

            new_word_counts: collections.Counter = collections.Counter()
            for word, freq in word_counts.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                        new_word.append(word[i] + word[i + 1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_counts[tuple(new_word)] = freq
            word_counts = new_word_counts

        # Populate vocabulary
        for pair in self.merges:
            self.vocab.add_token("".join(pair))
        for word in word_counts:
            for token in word:
                self.vocab.add_token(token)

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        words = text.split()
        tokens = []
        for word in words:
            word_tokens = list(word) + ["</w>"]
            for merge in self.merges:
                i = 0
                new_tokens = []
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i + 1]) == merge:
                        new_tokens.append(word_tokens[i] + word_tokens[i + 1])
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens
            tokens.extend(word_tokens)
        return tokens

    def _reconstruct_text(self, tokens: List[str]) -> str:
        text = "".join(tokens).replace("</w>", " ")
        return text.strip()


class WordPieceTokenizer(BaseTokenizer):
    """WordPiece subword tokenizer with '##' prefix for continuation subwords."""

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        tokens = []
        for word in words:
            if word in self.vocab:
                tokens.append(word)
                continue

            subwords = []
            start = 0
            is_bad = False
            while start < len(word):
                end = len(word)
                cur_substr = None
                while start < end:
                    substr = word[start:end]
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                subwords.append(cur_substr)
                start = end

            if is_bad:
                tokens.append(UNK_TOKEN)
            else:
                tokens.extend(subwords)
        return tokens


class SentencePieceTokenizer(BPETokenizer):
    """SentencePiece-style subword tokenizer using whitespace prefix ' '."""

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        text_with_prefix = " " + text.replace(" ", " ")
        return super().tokenize(text_with_prefix)

    def _reconstruct_text(self, tokens: List[str]) -> str:
        return "".join(tokens).replace(" ", " ").strip()


class CustomTokenizer(BaseTokenizer):
    """Custom tokenizer wrapper allowing user-defined tokenize and decode functions."""

    def __init__(self, tokenize_fn: Any = None, decode_fn: Any = None, vocab: Optional[Vocabulary] = None) -> None:
        super().__init__(vocab)
        self.tokenize_fn = tokenize_fn or (lambda text: text.split())
        self.decode_fn = decode_fn or (lambda tokens: " ".join(tokens))

    def tokenize(self, text: str) -> List[str]:
        return self.tokenize_fn(text)

    def _reconstruct_text(self, tokens: List[str]) -> str:
        return self.decode_fn(tokens)


class Tokenizer:
    """Factory and unified wrapper for all tokenization strategies in CarvusTrain."""

    @staticmethod
    def create(strategy: str = "word", vocab: Optional[Vocabulary] = None, **kwargs: Any) -> BaseTokenizer:
        """Create a tokenizer instance based on strategy name.

        Args:
            strategy: One of 'word', 'char', 'sentence', 'bpe', 'wordpiece', 'sentencepiece', 'custom'.
            vocab: Optional custom Vocabulary instance.
            **kwargs: Extra parameters for specific tokenizers.

        Returns:
            BaseTokenizer subclass instance.
        """
        strat = strategy.lower().strip()
        if strat in ("word", "default"):
            return WordTokenizer(vocab=vocab)
        elif strat == "char":
            return CharTokenizer(vocab=vocab)
        elif strat == "sentence":
            return SentenceTokenizer(vocab=vocab)
        elif strat == "bpe":
            return BPETokenizer(vocab=vocab, **kwargs)
        elif strat == "wordpiece":
            return WordPieceTokenizer(vocab=vocab)
        elif strat == "sentencepiece":
            return SentencePieceTokenizer(vocab=vocab, **kwargs)
        elif strat == "custom":
            return CustomTokenizer(vocab=vocab, **kwargs)
        else:
            raise TokenizerError(f"Unsupported tokenization strategy: '{strategy}'")
