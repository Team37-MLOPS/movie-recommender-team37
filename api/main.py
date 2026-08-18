from __future__ import annotations

import logging
import time
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Query, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from movie_recommender.config import load_settings
from movie_recommender.monitoring.logging import configure_logging
from movie_recommender.serving.repository import RecommendationRepository

configure_logging()
logger = logging.getLogger("movie_recommender.api")

settings = load_settings()

REQUEST_COUNT = Counter(
    "movie_recommender_api_requests_total",
    "Total API requests",
    ["endpoint", "status"],
)
PREDICTION_COUNT = Counter(
    "movie_recommender_predictions_total",
    "Total recommendation requests",
    ["fallback"],
)
RECOMMENDATIONS_RETURNED = Counter(
    "movie_recommender_recommendations_returned_total",
    "Total recommendation items returned",
    ["fallback"],
)
PREDICTION_SCORE = Histogram(
    "movie_recommender_prediction_score",
    "Distribution of recommendation scores served by the API",
    ["fallback"],
    buckets=(0, 1, 2, 3, 4, 5, 10, 20, 50, 100, float("inf")),
)
PREDICTION_RELEASE_YEAR = Counter(
    "movie_recommender_prediction_release_year_total",
    "Recommendation item release-year buckets served by the API",
    ["fallback", "year_bucket"],
)
REQUEST_LATENCY = Histogram(
    "movie_recommender_api_request_latency_seconds",
    "API request latency in seconds",
    ["endpoint"],
)


class RecommendationItem(BaseModel):
    movie_id: int
    title: str
    genres: str
    score: float
    rank: int
    release_year: int | None = None


class RecommendationResponse(BaseModel):
    user_id: int
    k: int
    fallback: bool
    recommendations: list[RecommendationItem]


@lru_cache
def get_repository() -> RecommendationRepository:
    return RecommendationRepository(
        settings.paths.recommendation_path,
        settings.paths.popularity_path,
    )


def release_year_bucket(release_year: int | None) -> str:
    if release_year is None:
        return "unknown"
    if release_year < 1980:
        return "pre_1980"
    if release_year < 1990:
        return "1980s"
    if release_year < 2000:
        return "1990s"
    if release_year < 2010:
        return "2000s"
    if release_year < 2020:
        return "2010s"
    return "2020s"


def observe_prediction_metrics(items: list[RecommendationItem], fallback: bool) -> None:
    fallback_label = str(fallback).lower()
    RECOMMENDATIONS_RETURNED.labels(fallback=fallback_label).inc(len(items))
    for item in items:
        PREDICTION_SCORE.labels(fallback=fallback_label).observe(item.score)
        PREDICTION_RELEASE_YEAR.labels(
            fallback=fallback_label,
            year_bucket=release_year_bucket(item.release_year),
        ).inc()


app = FastAPI(title="Movie Recommender APIs", version="0.1.0")
REPOSITORY_DEPENDENCY = Depends(get_repository)
K_QUERY = Query(default=settings.api.default_k, ge=1, le=settings.api.max_k)


@app.get("/health")
def health(repo: RecommendationRepository = REPOSITORY_DEPENDENCY) -> dict[str, str]:
    status = "ok" if repo.ready else "model_artifacts_missing"
    REQUEST_COUNT.labels(endpoint="/health", status=status).inc()
    return {"status": status}


@app.get("/ready")
def ready(repo: RecommendationRepository = REPOSITORY_DEPENDENCY) -> dict[str, str]:
    if not repo.ready:
        REQUEST_COUNT.labels(endpoint="/ready", status="not_ready").inc()
        raise HTTPException(status_code=503, detail="Recommendation artifacts are not available")
    REQUEST_COUNT.labels(endpoint="/ready", status="ready").inc()
    return {"status": "ready"}


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def recommendations(
    user_id: int,
    k: int = K_QUERY,
    repo: RecommendationRepository = REPOSITORY_DEPENDENCY,
) -> RecommendationResponse:
    start = time.perf_counter()
    endpoint = "/recommendations/{user_id}"
    try:
        if not repo.ready:
            REQUEST_COUNT.labels(endpoint=endpoint, status="not_ready").inc()
            raise HTTPException(
                status_code=503,
                detail="Recommendation artifacts are not available",
            )
        items, fallback = repo.recommend(user_id=user_id, k=k)
        PREDICTION_COUNT.labels(fallback=str(fallback).lower()).inc()
        response_items = [
            RecommendationItem.model_validate(item, from_attributes=True) for item in items
        ]
        observe_prediction_metrics(response_items, fallback)
        REQUEST_COUNT.labels(endpoint=endpoint, status="ok").inc()
        logger.info(
            "recommendations_served",
            extra={"user_id": user_id, "k": k, "fallback": fallback},
        )
        return RecommendationResponse(
            user_id=user_id,
            k=k,
            fallback=fallback,
            recommendations=response_items,
        )
    finally:
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
