"""FastAPI application for X-ray classification."""
from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.models.gradcam import get_gradcam_for_model
from src.models.inference import ModelInference, get_pretrained_model, load_model_from_weights
from src.utils.preprocessing import preprocess_image, validate_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "xray_api_requests_total",
    "Total number of requests",
    ["endpoint", "method"],
)
REQUEST_LATENCY = Histogram(
    "xray_api_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
)
PREDICTION_COUNT = Counter(
    "xray_predictions_total",
    "Total number of predictions",
    ["predicted_class"],
)
ERROR_COUNT = Counter(
    "xray_api_errors_total",
    "Total number of errors",
    ["endpoint", "error_type"],
)

# Global variables for model
model_inference: Optional[ModelInference] = None
gradcam_explainer = None
model_name = "vgg19"  # Default model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    global model_inference, gradcam_explainer, model_name
    
    # Skip model loading in tests or when explicitly disabled
    import os
    if os.getenv("SKIP_MODEL_LOADING") == "1":
        logger.info("Skipping model loading (test mode)")
        yield
        return
    
    logger.info("Loading model...")
    try:
        # Try to load pretrained model
        # In production, you would load your trained weights here
        model = get_pretrained_model(
            model_name=model_name,
            weights="imagenet",
            input_shape=(150, 150, 3),
            num_classes=1,
        )
        
        # Try to load custom weights if available
        weights_path = Path("models") / f"{model_name}_best.weights.h5"
        if weights_path.exists():
            model = load_model_from_weights(model, weights_path)
            logger.info(f"Loaded custom weights from {weights_path}")
        else:
            logger.warning(f"No custom weights found at {weights_path}. Using ImageNet weights.")
        
        model_inference = ModelInference(model, class_names=["NORMAL", "PNEUMONIA"])
        gradcam_explainer = get_gradcam_for_model(model, model_type=model_name)
        
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="X-Ray Classification API",
    description="API for chest X-ray classification using deep learning",
    version="1.0.0",
    lifespan=lifespan,
)


# Pydantic models
class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predicted_class: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., description="Confidence score for prediction")
    probabilities: Dict[str, float] = Field(..., description="Probabilities for all classes")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: str = Field(..., description="Name of the loaded model")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "X-Ray Classification API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "explain": "/explain",
            "health": "/health",
            "metrics": "/metrics",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    REQUEST_COUNT.labels(endpoint="/health", method="GET").inc()
    
    return HealthResponse(
        status="healthy" if model_inference is not None else "unhealthy",
        model_loaded=model_inference is not None,
        model_name=model_name,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """Predict chest X-ray classification.
    
    Args:
        file: X-ray image file (JPEG, PNG)
        
    Returns:
        Prediction results with class probabilities
    """
    REQUEST_COUNT.labels(endpoint="/predict", method="POST").inc()
    start_time = time.time()
    
    try:
        # Validate model is loaded
        if model_inference is None:
            ERROR_COUNT.labels(endpoint="/predict", error_type="model_not_loaded").inc()
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Read and validate image
        try:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            ERROR_COUNT.labels(endpoint="/predict", error_type="invalid_image").inc()
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Validate image
        if not validate_image(image):
            ERROR_COUNT.labels(endpoint="/predict", error_type="invalid_image_format").inc()
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Preprocess image
        processed_image = preprocess_image(image, target_size=(150, 150), normalize=True)
        
        # Make prediction
        inference_start = time.time()
        predicted_class, confidence = model_inference.get_prediction_label(processed_image)
        probabilities = model_inference.predict(processed_image)
        inference_time = (time.time() - inference_start) * 1000  # Convert to ms
        
        # Update metrics
        PREDICTION_COUNT.labels(predicted_class=predicted_class).inc()
        
        # Record latency
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            inference_time_ms=inference_time,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT.labels(endpoint="/predict", error_type="internal_error").inc()
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/explain", tags=["Explainability"])
async def explain(
    file: UploadFile = File(...),
    alpha: float = 0.4,
):
    """Generate Grad-CAM heatmap for X-ray image.
    
    Args:
        file: X-ray image file (JPEG, PNG)
        alpha: Transparency of the heatmap overlay (0.0 to 1.0)
        
    Returns:
        Overlaid image with Grad-CAM heatmap
    """
    REQUEST_COUNT.labels(endpoint="/explain", method="POST").inc()
    start_time = time.time()
    
    try:
        # Validate model is loaded
        if model_inference is None or gradcam_explainer is None:
            ERROR_COUNT.labels(endpoint="/explain", error_type="model_not_loaded").inc()
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Validate alpha parameter
        if not 0.0 <= alpha <= 1.0:
            raise HTTPException(status_code=400, detail="Alpha must be between 0.0 and 1.0")
        
        # Read and validate image
        try:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            ERROR_COUNT.labels(endpoint="/explain", error_type="invalid_image").inc()
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Validate image
        if not validate_image(image):
            ERROR_COUNT.labels(endpoint="/explain", error_type="invalid_image_format").inc()
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Keep original image for visualization
        original_image = image.copy()
        original_image = original_image.resize((150, 150))
        original_array = np.array(original_image)
        
        # Preprocess image for model inference
        processed_image = preprocess_image(image, target_size=(150, 150), normalize=True)
        
        # Generate heatmap using processed image
        heatmap = gradcam_explainer.generate_heatmap(processed_image)
        
        # Overlay on original (non-normalized) image for better visualization
        overlaid = gradcam_explainer.overlay_heatmap(heatmap, original_array, alpha)
        
        # Convert to PIL Image and return
        overlaid_pil = Image.fromarray(overlaid)
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        overlaid_pil.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Record latency
        REQUEST_LATENCY.labels(endpoint="/explain").observe(time.time() - start_time)
        
        return StreamingResponse(buffer, media_type="image/png")
    
    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT.labels(endpoint="/explain", error_type="internal_error").inc()
        logger.error(f"Explanation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return StreamingResponse(
        iter([generate_latest()]),
        media_type="text/plain",
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
