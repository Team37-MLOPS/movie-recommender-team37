from fastapi.testclient import TestClient

from api.main import app, get_repository


class FakeRepository:
    ready = True

    def recommend(self, user_id: int, k: int):
        item = type(
            "Recommendation",
            (),
            {
                "movie_id": 1,
                "title": "Toy Story",
                "genres": "Animation",
                "score": 5.0,
                "rank": 1,
                "release_year": 1995,
            },
        )()
        return [item], False


def test_recommendations_endpoint() -> None:
    app.dependency_overrides.clear()
    get_repository.cache_clear()
    app.dependency_overrides[get_repository] = lambda: FakeRepository()
    client = TestClient(app)

    response = client.get("/recommendations/1?k=1")

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 1
    assert body["recommendations"][0]["movie_id"] == 1


def test_metrics_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "movie_recommender_api_requests_total" in response.text
