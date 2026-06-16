"""
Integration tests for the FastAPI inference server.
These tests mock the model loader to avoid needing a real trained model.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.fixture(scope="module")
def mock_model():
    """Return a mock model that produces random scores."""
    model = MagicMock()
    # Simulate predict() returning an array of scores
    model.predict.side_effect = lambda inputs, **kwargs: np.random.rand(len(inputs[0])).astype(np.float32)
    return model


@pytest.fixture(scope="module")
def client(mock_model):
    """Build a test client with a mocked ModelLoader."""
    with patch("serving.model_loader.ModelLoader.get_instance") as mock_loader_cls:
        mock_loader = MagicMock()
        mock_loader.is_loaded = True
        mock_loader.version = "test-v1"
        mock_loader.predict.return_value = np.random.rand(200).astype(np.float32)
        mock_loader_cls.return_value = mock_loader

        from serving.server import app

        # Inject app state manually
        app.state.all_movie_idxs = np.arange(200, dtype=np.int32)
        app.state.movies_df = None
        app.state.movie_encoder = None

        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "uptime_seconds" in data


class TestModelInfo:
    def test_model_info_returns_200(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_schema(self, client):
        data = client.get("/model/info").json()
        assert "model_version" in data
        assert "stage" in data


class TestRecommendationsEndpoint:
    def test_recommend_returns_200(self, client):
        response = client.post("/recommend", json={"user_id": 1, "top_k": 5})
        assert response.status_code == 200

    def test_recommend_correct_count(self, client):
        response = client.post("/recommend", json={"user_id": 1, "top_k": 7})
        data = response.json()
        assert len(data["recommendations"]) == 7

    def test_recommend_schema(self, client):
        data = client.post("/recommend", json={"user_id": 1, "top_k": 3}).json()
        assert "user_id" in data
        assert "recommendations" in data
        assert "model_version" in data
        assert "latency_ms" in data

    def test_recommend_invalid_user(self, client):
        response = client.post("/recommend", json={"user_id": -1, "top_k": 5})
        assert response.status_code == 422

    def test_recommend_invalid_top_k(self, client):
        response = client.post("/recommend", json={"user_id": 1, "top_k": 0})
        assert response.status_code == 422


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        response = client.post("/predict", json={"user_id": 1, "movie_id": 10})
        assert response.status_code == 200

    def test_predict_schema(self, client):
        data = client.post("/predict", json={"user_id": 1, "movie_id": 10}).json()
        assert "predicted_rating" in data
        assert "confidence" in data
        assert "model_version" in data

    def test_predict_rating_in_range(self, client):
        data = client.post("/predict", json={"user_id": 1, "movie_id": 10}).json()
        assert 1.0 <= data["predicted_rating"] <= 5.0
