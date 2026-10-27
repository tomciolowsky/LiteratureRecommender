import pytest
from fastapi import status
from fastapi.testclient import TestClient

from literature_recommender.api.main import app


@pytest.fixture(scope="module")
def client():
    """
    Initializes TestClient within the FastAPI lifespan context manager.
    This ensures app.state.engine is initialized once for the entire test module.
    """
    with TestClient(app) as c:
        yield c


def test_health_info(client: TestClient):
    """
    Test the /health/info endpoint.
    """
    response = client.get("/health/info")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "literature-recommender-api"


def test_health_benchmark(client: TestClient):
    """
    Test the /health/benchmark endpoint.
    """
    response = client.get("/health/benchmark")
    assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED


def test_database_status(client: TestClient):
    """
    Test the /database/status endpoint.
    """
    response = client.get("/database/status")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "indexed_articles" in data
    assert "pending_articles" in data
    assert data["indexed_articles"] >= 0
    assert data["pending_articles"] >= 0


def _test_recommend(response, payload):
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["query_urls"] == payload["urls"]
    assert data["method"] == payload["method"]
    assert data["count"] == payload["top_k"]

    first_rec = data["recommendations"][0]
    expected_fields = {
        "title",
        "url",
        "final_score",
        "dense_score",
        "tfidf_score",
        "summary",
    }
    assert expected_fields.issubset(first_rec.keys())

    scores = [r["final_score"] for r in data["recommendations"]]
    assert scores == sorted(scores, reverse=True)


def test_recommend_single_url(client: TestClient):
    """
    Test the /recommend endpoint with a single URL.
    """
    payload = {
        "urls": ["https://en.wikipedia.org/wiki/Franz_Kafka"],
        "top_k": 3,
        "method": "hybrid",
        "alpha": 0.7
    }
    response = client.post("/recommend/", json=payload)
    _test_recommend(response, payload)


def test_recommend_multiple_urls(client: TestClient):
    """
    Test the /recommend endpoint with multiple URLs.
    """
    payload = {
        "urls": [
            "https://en.wikipedia.org/wiki/Jean-Paul_Sartre",
            "https://en.wikipedia.org/wiki/Albert_Camus",
        ],
        "top_k": 2,
        "method": "hybrid",
        "alpha": 0.5
    }
    response = client.post("/recommend/", json=payload)
    _test_recommend(response, payload)


def test_recommend_validation_error_empty_urls(client: TestClient):
    """
    Test the /recommend endpoint with an empty list of URLs.
    """
    payload = {"urls": []}
    response = client.post("/recommend/", json=payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_recommend_validation_error_invalid_method(client: TestClient):
    """
    Test the /recommend endpoint with an invalid recommendation method.
    """
    payload = {
        "urls": ["https://en.wikipedia.org/wiki/Franz_Kafka"],
        "method": "random_method",
    }
    response = client.post("/recommend/", json=payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_recommend_validation_error_invalid_alpha(client: TestClient):
    """
    Test the /recommend endpoint with an invalid alpha value (out of range).
    """
    payload = {
        "urls": ["https://en.wikipedia.org/wiki/Franz_Kafka"],
        "alpha": 2,
    }
    response = client.post("/recommend/", json=payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_recommend_validation_error_invalid_url_format(client: TestClient):
    """
    Test the /recommend endpoint with an invalid URL format.
    """
    payload = {
        "urls": ["not_a_valid_url_string"]
    }
    response = client.post("/recommend/", json=payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_database_reindex_execution(client: TestClient):
    """
    Test the /database/reindex endpoint.
    """
    response = client.post("/database/reindex")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "articles_added" in data
    assert "total_indexed" in data
    assert "message" in data
    assert data["articles_added"] >= 0 
    assert data["total_indexed"] >= 0 