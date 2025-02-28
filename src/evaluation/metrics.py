from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import average_precision_score


def mean_reciprocal_rank(relevance: np.ndarray, k: Optional[int] = None) -> float:
    """
    Calculate Mean Reciprocal Rank.

    Args:
        relevance: Binary relevance scores (1 for relevant, 0 for irrelevant)
        k: Optional cutoff for evaluation

    Returns:
        Mean Reciprocal Rank score
    """
    if k is not None:
        relevance = relevance[:k]

    ranks = np.where(relevance == 1)[0]
    if len(ranks) == 0:
        return 0.0

    return 1.0 / (ranks[0] + 1)


def precision_at_k(relevance: np.ndarray, k: int) -> float:
    """
    Calculate Precision@k.

    Args:
        relevance: Binary relevance scores (1 for relevant, 0 for irrelevant)
        k: Cutoff for evaluation

    Returns:
        Precision@k score
    """
    relevance = relevance[:k]
    return np.mean(relevance)


def recall_at_k(relevance: np.ndarray, k: int, n_relevant: int) -> float:
    """
    Calculate Recall@k.

    Args:
        relevance: Binary relevance scores (1 for relevant, 0 for irrelevant)
        k: Cutoff for evaluation
        n_relevant: Total number of relevant items

    Returns:
        Recall@k score
    """
    relevance = relevance[:k]
    return np.sum(relevance) / n_relevant


def mean_average_precision(relevance: np.ndarray, k: Optional[int] = None) -> float:
    """
    Calculate Mean Average Precision.

    Args:
        relevance: Binary relevance scores (1 for relevant, 0 for irrelevant)
        k: Optional cutoff for evaluation

    Returns:
        MAP score
    """
    if k is not None:
        relevance = relevance[:k]

    return average_precision_score(relevance, np.ones_like(relevance))


def ndcg_at_k(
    relevance: np.ndarray, k: int, gains: Optional[Dict[int, float]] = None
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain@k.

    Args:
        relevance: Relevance scores
        k: Cutoff for evaluation
        gains: Optional mapping of relevance levels to gain values

    Returns:
        NDCG@k score
    """
    if gains is not None:
        relevance = np.array([gains.get(r, 0.0) for r in relevance])

    relevance = relevance[:k]
    ideal_relevance = -np.sort(-relevance)

    def dcg(scores: np.ndarray) -> float:
        return np.sum((2**scores - 1) / np.log2(np.arange(2, len(scores) + 2)))

    actual_dcg = dcg(relevance)
    ideal_dcg = dcg(ideal_relevance)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def click_through_rate(clicks: np.ndarray, impressions: np.ndarray) -> float:
    """
    Calculate Click-Through Rate.

    Args:
        clicks: Number of clicks per item
        impressions: Number of impressions per item

    Returns:
        CTR score
    """
    return np.sum(clicks) / np.sum(impressions)


class RankingMetrics:
    """Collection of ranking metrics for evaluation."""

    def __init__(
        self,
        relevance: np.ndarray,
        n_relevant: Optional[int] = None,
        gains: Optional[Dict[int, float]] = None,
    ):
        self.relevance = relevance
        self.n_relevant = n_relevant or np.sum(relevance)
        self.gains = gains

    def compute_metrics(
        self, k: Optional[int] = None, metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute specified ranking metrics.

        Args:
            k: Optional cutoff for evaluation
            metrics: List of metric names to compute

        Returns:
            Dictionary of metric names to scores
        """
        if metrics is None:
            metrics = ["mrr", "map", "ndcg"]

        results = {}

        for metric in metrics:
            if metric == "mrr":
                results["mrr"] = mean_reciprocal_rank(self.relevance, k)
            elif metric == "precision":
                results[f"precision@{k}"] = precision_at_k(self.relevance, k)
            elif metric == "recall":
                results[f"recall@{k}"] = recall_at_k(self.relevance, k, self.n_relevant)
            elif metric == "map":
                results["map"] = mean_average_precision(self.relevance, k)
            elif metric == "ndcg":
                results[f"ndcg@{k}"] = ndcg_at_k(self.relevance, k, self.gains)

        return results
