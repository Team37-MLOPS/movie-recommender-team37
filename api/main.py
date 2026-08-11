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


app = FastAPI(title="Movie Recommender Team37", version="0.1.0")
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
        REQUEST_COUNT.labels(endpoint=endpoint, status="ok").inc()
        logger.info(
            "recommendations_served",
            extra={"user_id": user_id, "k": k, "fallback": fallback},
        )
        return RecommendationResponse(
            user_id=user_id,
            k=k,
            fallback=fallback,
            recommendations=[
                RecommendationItem.model_validate(item, from_attributes=True) for item in items
            ],
        )
    finally:
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
