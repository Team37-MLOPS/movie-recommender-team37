from unittest.mock import patch

import pandas as pd

from movie_recommender.serving.champion_repository import ChampionRepository
from movie_recommender.serving.repository import RecommendationRepository


def test_recommendation_repository_returns_user_recommendations(tmp_path) -> None:
    recs_path = tmp_path / "recommendations.parquet"
    popular_path = tmp_path / "popular.parquet"

    pd.DataFrame(
        [
            {
                "user_id": 1,
                "movie_id": 10,
                "rank": 1,
                "score": 4.8,
                "clean_title": "Toy Story",
                "genres": "Animation|Children's|Comedy",
                "release_year": 1995,
            }
        ]
    ).to_parquet(recs_path)
    pd.DataFrame(
        [
            {
                "movie_id": 20,
                "score": 4.5,
                "clean_title": "Popular Movie",
                "genres": "Drama",
                "release_year": 1999,
            }
        ]
    ).to_parquet(popular_path)

    repo = RecommendationRepository(recs_path, popular_path)
    items, fallback = repo.recommend(user_id=1, k=10)

    assert fallback is False
    assert items[0].movie_id == 10
    assert items[0].rank == 1


def test_recommendation_repository_falls_back_to_popularity(tmp_path) -> None:
    recs_path = tmp_path / "recommendations.parquet"
    popular_path = tmp_path / "popular.parquet"

    pd.DataFrame(
        [
            {
                "movie_id": 20,
                "score": 4.5,
                "clean_title": "Popular Movie",
                "genres": "Drama",
                "release_year": 1999,
            }
        ]
    ).to_parquet(popular_path)

    repo = RecommendationRepository(recs_path, popular_path)
    items, fallback = repo.recommend(user_id=999, k=1)

    assert fallback is True
    assert items[0].movie_id == 20


def _write_fallback_parquet(tmp_path):
    recs_path = tmp_path / "recommendations.parquet"
    popular_path = tmp_path / "popular.parquet"

    pd.DataFrame(
        [
            {
                "movie_id": 20,
                "score": 4.5,
                "clean_title": "Popular Movie",
                "genres": "Drama",
                "release_year": 1999,
            }
        ]
    ).to_parquet(popular_path)
    pd.DataFrame(columns=["user_id", "movie_id", "rank", "score"]).to_parquet(recs_path)
    return recs_path, popular_path


def test_champion_repository_serves_from_registry_model(tmp_path) -> None:
    recs_path, popular_path = _write_fallback_parquet(tmp_path)

    fake_model = type("FakeModel", (), {"predict": lambda self, df: [[20]]})()
    with patch("mlflow.pyfunc.load_model", return_value=fake_model):
        repo = ChampionRepository(recs_path, popular_path)
        items, fallback = repo.recommend(user_id=1, k=1)

    assert fallback is False
    assert items[0].movie_id == 20
    assert items[0].title == "Popular Movie"


def test_champion_repository_falls_back_to_popularity_on_model_error(tmp_path) -> None:
    recs_path, popular_path = _write_fallback_parquet(tmp_path)

    def _raise(self, df):
        raise RuntimeError("cold-start user")

    fake_model = type("FakeModel", (), {"predict": _raise})()
    with patch("mlflow.pyfunc.load_model", return_value=fake_model):
        repo = ChampionRepository(recs_path, popular_path)
        items, fallback = repo.recommend(user_id=999, k=1)

    assert fallback is True
    assert items[0].movie_id == 20


def test_champion_repository_falls_back_when_registry_unavailable(tmp_path) -> None:
    recs_path, popular_path = _write_fallback_parquet(tmp_path)

    with patch("mlflow.pyfunc.load_model", side_effect=RuntimeError("registry unreachable")):
        repo = ChampionRepository(recs_path, popular_path)

    assert repo.ready
    items, fallback = repo.recommend(user_id=1, k=1)
    assert fallback is True
    assert items[0].movie_id == 20
