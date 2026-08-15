import pandas as pd

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
