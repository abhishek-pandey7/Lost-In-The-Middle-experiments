"""Metrics and scoring utilities."""
import re
import statistics
from typing import List, Dict, Any


def exact_match_score(prediction: str, target: str) -> float:
    """Binary exact-match score."""
    return 1.0 if target.lower() in prediction.lower() else 0.0


def numeric_match(prediction: str, target: float, tolerance: float = 0.5) -> float:
    """Check if any number in the prediction is within tolerance of target."""
    nums = re.findall(r"[\d,]+\.?\d*", prediction.replace(",", ""))
    if not nums:
        return 0.0
    for n in nums:
        try:
            if abs(float(n) - target) < tolerance:
                return 1.0
        except ValueError:
            continue
    return 0.0


def compute_accuracy(predictions: List[Dict[str, Any]], key: str = "correct") -> float:
    """Mean accuracy from a list of prediction records."""
    vals = [p[key] for p in predictions]
    return statistics.mean(vals) if vals else 0.0


def position_bias_index(positions: List[float], accuracies: List[float]) -> float:
    """
    Compute Position Bias Index (PBI):
    PBI = (acc_first + acc_last) / 2 - acc_middle
    Higher PBI = stronger U-shape (worse).
    """
    if len(positions) < 3:
        return 0.0
    mid_idx = len(positions) // 2
    edge_acc = (accuracies[0] + accuracies[-1]) / 2.0
    mid_acc = accuracies[mid_idx]
    return edge_acc - mid_acc
