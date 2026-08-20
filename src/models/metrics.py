"""Evaluation metrics for the recommender: rating-prediction accuracy
(RMSE, MAE) and top-K ranking quality (Precision@K, Recall@K, F1@K)."""
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
    scored.

    F1@K is the harmonic mean of precision and recall, computed per-user
    then averaged - this captures models that win on recall by casting a
    wide net (hurting precision) or win on precision by playing it safe
    with a couple of obvious hits (hurting recall), which recall or
    precision alone would miss.
    """
    precisions, recalls, f1s = [], [], []
    for user_id, relevant in relevant_items.items():
        if not relevant:
            continue
        recs = recommendations.get(user_id, [])[:k]
        if not recs:
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)
            continue
        hits = len(set(recs) & relevant)
        precision = hits / len(recs)
        recall = hits / len(relevant)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        f"precision_at_{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall_at_{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"f1_at_{k}": float(np.mean(f1s)) if f1s else 0.0,
    }

