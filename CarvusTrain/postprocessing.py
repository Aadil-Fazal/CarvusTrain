"""Text generation postprocessing, logit sampling, and text formatting strategies."""

import math
import random
from typing import List, Optional, Tuple, Union


class LogitProcessor:
    """Processes raw output logits using temperature, top-k, top-p, and repetition penalties."""

    @staticmethod
    def apply_temperature(logits: List[float], temperature: float = 1.0) -> List[float]:
        """Scale logits by temperature factor.

        Args:
            logits: Unnormalized log probabilities.
            temperature: Sampling temperature (>0.0). Lower values make distribution sharper.

        Returns:
            Temperature-scaled logits.
        """
        temp = max(1e-5, float(temperature))
        return [l / temp for l in logits]

    @staticmethod
    def apply_repetition_penalty(logits: List[float], generated_ids: List[int], penalty: float = 1.1) -> List[float]:
        """Apply repetition penalty to logits for previously generated tokens."""
        if penalty == 1.0 or not generated_ids:
            return logits

        logits_copy = list(logits)
        seen_ids = set(generated_ids)
        for idx in seen_ids:
            if idx < len(logits_copy):
                if logits_copy[idx] < 0:
                    logits_copy[idx] *= penalty
                else:
                    logits_copy[idx] /= penalty
        return logits_copy

    @staticmethod
    def sample_top_k_top_p(logits: List[float], top_k: int = 50, top_p: float = 0.9) -> int:
        """Sample token ID from logits using combined Top-K and Top-P (nucleus) sampling.

        Args:
            logits: Raw or temperature-scaled logits list.
            top_k: Keep highest k probable logits.
            top_p: Keep cumulative probability mass p.

        Returns:
            Sampled token ID integer.
        """
        # Convert logits to probabilities via softmax
        max_l = max(logits)
        exps = [math.exp(l - max_l) for l in logits]
        sum_exps = sum(exps)
        probs = [e / sum_exps for e in exps]

        indexed_probs = list(enumerate(probs))
        indexed_probs.sort(key=lambda x: x[1], reverse=True)

        # Apply Top-K filtering
        if top_k > 0:
            indexed_probs = indexed_probs[:top_k]

        # Apply Top-P (nucleus) filtering
        if top_p < 1.0:
            cum_prob = 0.0
            cutoff_idx = len(indexed_probs)
            for idx, (tok_id, prob) in enumerate(indexed_probs):
                cum_prob += prob
                if cum_prob >= top_p:
                    cutoff_idx = idx + 1
                    break
            indexed_probs = indexed_probs[:cutoff_idx]

        # Re-normalize filtered probabilities
        total_p = sum(p for _, p in indexed_probs)
        norm_probs = [p / total_p for _, p in indexed_probs]
        token_ids = [tok_id for tok_id, _ in indexed_probs]

        # Weighted random choice
        r = random.random()
        cum = 0.0
        for tok_id, p in zip(token_ids, norm_probs):
            cum += p
            if r <= cum:
                return tok_id

        return token_ids[0]


class TextPostprocessor:
    """Post-processes generated token text strings (truncation on stop tokens, whitespace formatting)."""

    def __init__(self, stop_sequences: Optional[List[str]] = None) -> None:
        self.stop_sequences = stop_sequences or ["\nUser:", "<eos>", "</s>"]

    def process(self, text: str) -> str:
        """Truncate and clean generated text."""
        if not text:
            return ""

        # Truncate at first stop sequence
        for stop_seq in self.stop_sequences:
            if stop_seq in text:
                text = text.split(stop_seq)[0]

        # Clean trailing spaces
        return text.strip()
