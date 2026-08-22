from __future__ import annotations

import logging
import os
from pathlib import Path

import mlflow.pyfunc
import pandas as pd

from movie_recommender.serving.repository import Recommendation, RecommendationRepository

logger = logging.getLogger("movie_recommender.serving.champion_repository")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH}")
CHAMPION_MODEL_URI = "models:/movie-recommender-prod@champion"


class ChampionRepository:
    """Serves recommendations from the MLflow-registered champion model
    (whichever tuned model family `movie_recommender.models.select_best` most recently
    crowned), falling back to the static popularity table - via a plain
    `RecommendationRepository` over the same parquet artifacts - when the
    champion model can't be loaded at all, or fails on a specific request
    (e.g. a cold-start user/movie id the underlying model has no learned
    state for).

    Loading the champion model is a single `mlflow.pyfunc.load_model` call
    made once at construction time, not per request; callers that want to
    pick up a newly-registered champion without restarting the process
    should drop this instance (e.g. via `get_repository.cache_clear()`)
    and let a fresh one be built on the next request.
    """

    def __init__(self, recommendation_path: Path, popularity_path: Path) -> None:
        self._fallback = RecommendationRepository(recommendation_path, popularity_path)
        self._model = self._load_champion()

    @staticmethod
    def _load_champion():
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            return mlflow.pyfunc.load_model(CHAMPION_MODEL_URI)
        except Exception as exc:  # noqa: BLE001 - registry/model absence must not crash the app
            logger.warning("Champion model unavailable, falling back to popularity table: %s", exc)
            return None

    @property
    def ready(self) -> bool:
        return self._model is not None or self._fallback.ready

    def recommend(self, user_id: int, k: int) -> tuple[list[Recommendation], bool]:
        if self._model is not None:
            try:
                movie_ids = self._model.predict(pd.DataFrame([{"user_id": user_id, "k": k}]))[0]
            except Exception as exc:  # noqa: BLE001 - any per-request model failure falls back
                logger.warning("Champion model prediction failed for user_id=%s, falling back: %s", user_id, exc)
                movie_ids = []
            if movie_ids:
                return self._to_recommendations(movie_ids), False

        return self._fallback.recommend(user_id, k)

    def _to_recommendations(self, movie_ids: list[int]) -> list[Recommendation]:
        n = len(movie_ids)
        items: list[Recommendation] = []
        for rank, movie_id in enumerate(movie_ids, start=1):
            metadata = self._fallback.movie_metadata(int(movie_id))
            items.append(
                Recommendation(
                    movie_id=int(movie_id),
                    title=metadata.get("title", ""),
                    genres=metadata.get("genres", ""),
                    # The champion pyfunc model returns a ranked movie_id list, not
                    # per-item scores, so this is a synthetic rank-derived score
                    # (highest for rank 1) rather than a learned prediction score.
                    score=round((n - rank + 1) / n, 4),
                    rank=rank,
                    release_year=metadata.get("release_year"),
                )
            )
        return items
