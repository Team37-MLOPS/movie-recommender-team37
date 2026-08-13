"""Evaluation metrics for the recommender: rating-prediction accuracy
(RMSE, MAE) and top-K ranking quality (Precision@K, Recall@K)."""
from typing import Dict, List, Sequence

import numpy as np


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def precision_recall_at_k(
    recommendations: Dict[int, List[int]], relevant_items: Dict[int, set], k: int = 10
) -> Dict[str, float]:
    """recommendations: user_id -> ranked top-K movie_ids.
    relevant_items: user_id -> set of movie_ids the user actually liked in
    the held-out test set. Only users with at least one relevant item are
    scored."""
    precisions, recalls = [], []
    for user_id, relevant in relevant_items.items():
        if not relevant:
            continue
        recs = recommendations.get(user_id, [])[:k]
        if not recs:
            precisions.append(0.0)
            recalls.append(0.0)
            continue
        hits = len(set(recs) & relevant)
        precisions.append(hits / len(recs))
        recalls.append(hits / len(relevant))

    return {
        f"precision_at_{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall_at_{k}": float(np.mean(recalls)) if recalls else 0.0,
    }
