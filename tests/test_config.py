from movie_recommender.config import load_settings


def test_load_settings() -> None:
    settings = load_settings()
    assert settings.project_name == "movie-recommender-team37"
    assert settings.training.top_k > 0
    assert settings.paths.dataset_zip.name == "ml-1m.zip"
