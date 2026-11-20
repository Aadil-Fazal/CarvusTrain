"""Performance evaluation metrics for classification, generation, and language modeling."""

import collections
import math
from typing import Any, Dict, List, Optional, Union


def compute_accuracy(predictions: List[Any], targets: List[Any]) -> float:
    """Calculate classification accuracy.

    Args:
        predictions: Predicted class labels or sequence tokens.
        targets: Target class labels or sequence tokens.

    Returns:
        Accuracy score between 0.0 and 1.0.
    """
    if not predictions or len(predictions) != len(targets):
        return 0.0

    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    return correct / len(predictions)


def compute_precision_recall_f1(predictions: List[int], targets: List[int], positive_label: int = 1) -> Dict[str, float]:
    """Calculate Precision, Recall, and F1-Score for binary/multiclass evaluation.

    Args:
        predictions: List of predicted labels.
        targets: List of ground-truth labels.
        positive_label: Class label treated as positive.

    Returns:
        Dictionary with keys 'precision', 'recall', 'f1_score'.
    """
    tp = sum(1 for p, t in zip(predictions, targets) if p == positive_label and t == positive_label)
    fp = sum(1 for p, t in zip(predictions, targets) if p == positive_label and t != positive_label)
    fn = sum(1 for p, t in zip(predictions, targets) if p != positive_label and t == positive_label)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def compute_bleu(hypothesis: List[str], references: List[List[str]], max_n: int = 4) -> float:
    """Compute BLEU score for candidate hypothesis text against reference texts.

    Args:
        hypothesis: List of candidate tokens.
        references: List of reference token lists.
        max_n: Maximum n-gram order.

    Returns:
        BLEU score between 0.0 and 1.0.
    """
    if not hypothesis or not references:
        return 0.0

    hyp_len = len(hypothesis)
    ref_lens = [len(r) for r in references]
    closest_ref_len = min(ref_lens, key=lambda r: (abs(r - hyp_len), r))

    # Brevity Penalty
    if hyp_len > closest_ref_len:
        bp = 1.0
    elif hyp_len == 0:
        return 0.0
    else:
        bp = math.exp(1 - closest_ref_len / hyp_len)

    p_ns = []
    for n in range(1, max_n + 1):
        hyp_ngrams = collections.Counter(tuple(hypothesis[i : i + n]) for i in range(len(hypothesis) - n + 1))
        if not hyp_ngrams:
            p_ns.append(1e-8)
            continue

        max_ref_ngrams: collections.Counter = collections.Counter()
        for ref in references:
            ref_ngrams = collections.Counter(tuple(ref[i : i + n]) for i in range(len(ref) - n + 1))
            for ngram, count in ref_ngrams.items():
                max_ref_ngrams[ngram] = max(max_ref_ngrams[ngram], count)

        clipped_count = sum(min(count, max_ref_ngrams[ngram]) for ngram, count in hyp_ngrams.items())
        total_count = sum(hyp_ngrams.values())

        p_ns.append(clipped_count / total_count if total_count > 0 else 1e-8)

    # Geometric mean of n-gram precisions
    log_sum = sum(math.log(max(p, 1e-8)) for p in p_ns) / max_n
    return bp * math.exp(log_sum)


def compute_rouge_n(hypothesis: List[str], reference: List[str], n: int = 1) -> float:
    """Compute ROUGE-N recall score.

    Args:
        hypothesis: List of candidate tokens.
        reference: List of reference tokens.
        n: N-gram order.

    Returns:
        ROUGE-N recall score.
    """
    if not hypothesis or not reference:
        return 0.0

    ref_ngrams = collections.Counter(tuple(reference[i : i + n]) for i in range(len(reference) - n + 1))
    hyp_ngrams = collections.Counter(tuple(hypothesis[i : i + n]) for i in range(len(hypothesis) - n + 1))

    overlap = sum(min(count, hyp_ngrams[ngram]) for ngram, count in ref_ngrams.items())
    total_ref = sum(ref_ngrams.values())

    return overlap / total_ref if total_ref > 0 else 0.0


class MetricTracker:
    """Tracker for accumulating losses, accuracies, and timing metrics across epochs."""

    def __init__(self) -> None:
        self.metrics: Dict[str, List[float]] = collections.defaultdict(list)

    def update(self, name: str, value: float) -> None:
        """Record a metric value."""
        self.metrics[name].append(float(value))

    def get_average(self, name: str) -> float:
        """Get running average of recorded metric."""
        vals = self.metrics.get(name, [])
        return sum(vals) / len(vals) if vals else 0.0

    def get_latest(self, name: str) -> float:
        """Get latest recorded value for metric."""
        vals = self.metrics.get(name, [])
        return vals[-1] if vals else 0.0

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.metrics.clear()

    def summary(self) -> Dict[str, float]:
        """Return dictionary of average values for all tracked metrics."""
        return {k: sum(v) / len(v) if v else 0.0 for k, v in self.metrics.items()}
