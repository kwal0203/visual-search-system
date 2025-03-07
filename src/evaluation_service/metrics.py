from sklearn.metrics import average_precision_score
from typing import Dict, List, Optional, Union
import numpy as np
import warnings

warnings.simplefilter("ignore")


def mean_reciprocal_rank(
    relevance: np.ndarray, k: Optional[int] = None
) -> Union[float, np.ndarray]:
    """
    Calculate Mean Reciprocal Rank for single or multiple ranking lists.

    Args:
        relevance: Binary relevance scores (1 for relevant, 0 for irrelevant)
                  Shape: (n_queries, n_docs) or (n_docs,)
        k: Optional cutoff for evaluation

    Returns:
        Mean Reciprocal Rank score(s)
    """
    if k is not None:
        relevance = relevance[..., :k]

    if relevance.ndim == 1:
        ranks = np.where(relevance == 1)[0]
        if len(ranks) == 0:
            return 0.0
        return 1.0 / (ranks[0] + 1)

    # Handle batch of rankings
    ranks = [np.where(rel == 1)[0] for rel in relevance]
    mrr_scores = np.zeros(len(relevance))
    for i, rank in enumerate(ranks):
        if len(rank) > 0:
            mrr_scores[i] = 1.0 / (rank[0] + 1)
    return mrr_scores


def precision_at_k(relevance: np.ndarray, k: int) -> Union[float, np.ndarray]:
    """
    Calculate Precision@k for single or multiple ranking lists.

    Args:
        relevance: Binary relevance scores (1 for relevant, 0 for irrelevant)
                  Shape: (n_queries, n_docs) or (n_docs,)
        k: Cutoff for evaluation

    Returns:
        Precision@k score(s)
    """
    if relevance.ndim == 1:
        return np.mean(relevance[:k])
    return np.mean(relevance[..., :k], axis=1)


def recall_at_k(
    relevance: np.ndarray, k: int, n_relevant: Union[int, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate Recall@k for single or multiple ranking lists.

    Args:
        relevance: Binary relevance scores (1 for relevant, 0 for irrelevant)
                  Shape: (n_queries, n_docs) or (n_docs,)
        k: Cutoff for evaluation
        n_relevant: Total number of relevant items per query or single value

    Returns:
        Recall@k score(s)
    """
    if relevance.ndim == 1:
        return np.sum(relevance[:k]) / n_relevant

    recall_scores = np.sum(relevance[..., :k], axis=1)
    if isinstance(n_relevant, (int, float)):
        n_relevant = np.full(len(relevance), n_relevant)
    return recall_scores / n_relevant


def mean_average_precision(
    relevance: np.ndarray, k: Optional[int] = None
) -> Union[float, np.ndarray]:
    """
    Calculate Mean Average Precision for single or multiple ranking lists.

    Args:
        relevance: Binary relevance scores (1 for relevant, 0 for irrelevant)
                  Shape: (n_queries, n_docs) or (n_docs,)
        k: Optional cutoff for evaluation

    Returns:
        MAP score(s)
    """
    if k is not None:
        relevance = relevance[..., :k]

    if relevance.ndim == 1:
        return average_precision_score(relevance, np.ones_like(relevance))

    map_scores = np.zeros(len(relevance))
    for i, rel in enumerate(relevance):
        map_scores[i] = average_precision_score(rel, np.ones_like(rel))
        # print(f"  -- BANG: {map_scores[i]}")
    return map_scores


def ndcg_at_k(
    relevance: np.ndarray, k: int, gains: Optional[Dict[int, float]] = None
) -> Union[float, np.ndarray]:
    """
    Calculate Normalized Discounted Cumulative Gain@k for single or multiple ranking lists.

    Args:
        relevance: Relevance scores
                  Shape: (n_queries, n_docs) or (n_docs,)
        k: Cutoff for evaluation
        gains: Optional mapping of relevance levels to gain values

    Returns:
        NDCG@k score(s)
    """
    if gains is not None:
        if relevance.ndim == 1:
            relevance = np.array([gains.get(r, 0.0) for r in relevance])
        else:
            relevance = np.array(
                [[gains.get(r, 0.0) for r in rel] for rel in relevance]
            )

    def dcg(scores: np.ndarray) -> float:
        return np.sum((2**scores - 1) / np.log2(np.arange(2, len(scores) + 2)))

    if relevance.ndim == 1:
        relevance = relevance[:k]
        ideal_relevance = -np.sort(-relevance)
        actual_dcg = dcg(relevance)
        ideal_dcg = dcg(ideal_relevance)
        return 0.0 if ideal_dcg == 0 else actual_dcg / ideal_dcg

    ndcg_scores = np.zeros(len(relevance))
    for i, rel in enumerate(relevance):
        rel = rel[:k]
        ideal_rel = -np.sort(-rel)
        actual_dcg = dcg(rel)
        ideal_dcg = dcg(ideal_rel)
        ndcg_scores[i] = 0.0 if ideal_dcg == 0 else actual_dcg / ideal_dcg
    return ndcg_scores


def click_through_rate(clicks: List, impressions: List) -> Union[float, np.ndarray]:
    """
    Calculate Click-Through Rate for single or multiple ranking lists.

    Args:
        clicks: Number of clicks per item
               Shape: (n_queries, n_docs) or (n_docs,)
        impressions: Number of impressions per item
                    Shape: (n_queries, n_docs) or (n_docs,)

    Returns:
        CTR score(s)
    """
    clicks = np.array(clicks)
    impressions = np.array(impressions)
    if clicks.ndim == 1:
        return np.sum(clicks) / np.sum(impressions)
    return np.sum(clicks, axis=1) / np.sum(impressions, axis=1)


class RankingMetrics:
    """Collection of ranking metrics for evaluation."""

    def __init__(
        self,
        relevance: List,
        n_relevant: Optional[Union[int, np.ndarray]] = None,
        gains: Optional[Dict[int, float]] = None,
    ):
        """
        Initialize RankingMetrics.

        Args:
            relevance: Relevance scores. Shape: (n_queries, n_docs) or (n_docs,)
            n_relevant: Total number of relevant items per query or single value
            gains: Optional mapping of relevance levels to gain values
        """
        self.relevance = np.array(relevance)
        if n_relevant is None:
            if relevance.ndim == 1:
                self.n_relevant = np.sum(self.relevance)
            else:
                self.n_relevant = np.sum(self.relevance, axis=1)
        else:
            self.n_relevant = n_relevant
        self.gains = gains

    def compute_metrics(
        self, k: Optional[int] = None, metrics: Optional[List[str]] = None
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute specified ranking metrics.

        Args:
            k: Optional cutoff for evaluation
            metrics: List of metric names to compute

        Returns:
            Dictionary of metric names to scores. For batch evaluation,
            each score is an array of shape (n_queries,)
        """
        if metrics is None:
            metrics = ["mrr", "precision", "recall", "map", "ndcg"]

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
            else:
                raise ValueError(f"Metric {metric} not supported")

        return results

    def compute_average_metrics(
        self, k: Optional[int] = None, metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute average metrics across all queries.

        Args:
            k: Optional cutoff for evaluation
            metrics: List of metric names to compute

        Returns:
            Dictionary of metric names to average scores
        """
        results = self.compute_metrics(k, metrics)
        return {name: float(np.mean(score)) for name, score in results.items()}
