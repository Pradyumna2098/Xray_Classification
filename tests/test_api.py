"""Integration tests for FastAPI endpoints."""
import io
from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_model_inference():
    """Mock model inference."""
    with patch("api.main.model_inference") as mock:
        mock.predict.return_value = {"NORMAL": 0.3, "PNEUMONIA": 0.7}
        mock.get_prediction_label.return_value = ("PNEUMONIA", 0.7)
        yield mock


@pytest.fixture
def mock_gradcam():
    """Mock Grad-CAM explainer."""
    with patch("api.main.gradcam_explainer") as mock:
        heatmap = np.random.rand(150, 150)
        overlaid = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        mock.explain_prediction.return_value = (heatmap, overlaid)
        yield mock


@pytest.fixture
def test_image():
    """Create a test image."""
    image = Image.new("RGB", (200, 200), color="red")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check_healthy(self, client, mock_model_inference):
        """Test health check when model is loaded."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "model_name" in data
    
    def test_health_check_unhealthy(self, client):
        """Test health check when model is not loaded."""
        with patch("api.main.model_inference", None):
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["model_loaded"] is False


class TestPredictEndpoint:
    """Tests for prediction endpoint."""
    
    def test_predict_success(self, client, mock_model_inference, test_image):
        """Test successful prediction."""
        response = client.post(
            "/predict",
            files={"file": ("test.png", test_image, "image/png")},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "inference_time_ms" in data
        assert data["predicted_class"] == "PNEUMONIA"
        assert data["confidence"] == 0.7
    
    def test_predict_model_not_loaded(self, client, test_image):
        """Test prediction when model is not loaded."""
        with patch("api.main.model_inference", None):
            response = client.post(
                "/predict",
                files={"file": ("test.png", test_image, "image/png")},
            )
            
            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]
    
    def test_predict_invalid_image(self, client, mock_model_inference):
        """Test prediction with invalid image."""
        invalid_data = io.BytesIO(b"not an image")
        
        response = client.post(
            "/predict",
            files={"file": ("test.txt", invalid_data, "text/plain")},
        )
        
        assert response.status_code == 400
        assert "Invalid image file" in response.json()["detail"]
    
    def test_predict_no_file(self, client, mock_model_inference):
        """Test prediction without file."""
        response = client.post("/predict")
        
        assert response.status_code == 422  # Validation error


class TestExplainEndpoint:
    """Tests for explain endpoint."""
    
    def test_explain_success(self, client, mock_model_inference, mock_gradcam, test_image):
        """Test successful explanation."""
        response = client.post(
            "/explain",
            files={"file": ("test.png", test_image, "image/png")},
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
    
    def test_explain_with_alpha(self, client, mock_model_inference, mock_gradcam, test_image):
        """Test explanation with custom alpha."""
        response = client.post(
            "/explain?alpha=0.6",
            files={"file": ("test.png", test_image, "image/png")},
        )
        
        assert response.status_code == 200
    
    def test_explain_invalid_alpha(self, client, mock_model_inference, mock_gradcam, test_image):
        """Test explanation with invalid alpha."""
        response = client.post(
            "/explain?alpha=1.5",
            files={"file": ("test.png", test_image, "image/png")},
        )
        
        assert response.status_code == 400
        assert "Alpha must be between" in response.json()["detail"]
    
    def test_explain_model_not_loaded(self, client, test_image):
        """Test explanation when model is not loaded."""
        with patch("api.main.model_inference", None):
            with patch("api.main.gradcam_explainer", None):
                response = client.post(
                    "/explain",
                    files={"file": ("test.png", test_image, "image/png")},
                )
                
                assert response.status_code == 503


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""
    
    def test_metrics(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        
        # Check for Prometheus format
        content = response.text
        assert "xray_api_requests_total" in content or "HELP" in content
