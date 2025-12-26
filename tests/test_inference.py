"""Unit tests for model inference utilities."""
import numpy as np
import pytest
from unittest.mock import Mock, patch

from src.models.inference import ModelInference, get_pretrained_model


class TestModelInference:
    """Tests for ModelInference class."""
    
    def test_initialization(self):
        """Test ModelInference initialization."""
        model = Mock()
        inference = ModelInference(model, class_names=["A", "B"])
        
        assert inference.model == model
        assert inference.class_names == ["A", "B"]
    
    def test_default_class_names(self):
        """Test default class names."""
        model = Mock()
        inference = ModelInference(model)
        
        assert inference.class_names == ["NORMAL", "PNEUMONIA"]
    
    def test_predict_binary_sigmoid(self):
        """Test prediction with binary classification (sigmoid)."""
        model = Mock()
        model.predict.return_value = np.array([[0.7]])
        
        inference = ModelInference(model, class_names=["NORMAL", "PNEUMONIA"])
        image = np.zeros((1, 150, 150, 3))
        
        predictions = inference.predict(image)
        
        assert "NORMAL" in predictions
        assert "PNEUMONIA" in predictions
        assert abs(predictions["NORMAL"] - 0.3) < 0.01
        assert abs(predictions["PNEUMONIA"] - 0.7) < 0.01
    
    def test_predict_multiclass_softmax(self):
        """Test prediction with multiclass classification (softmax)."""
        model = Mock()
        model.predict.return_value = np.array([[0.3, 0.7]])
        
        inference = ModelInference(model, class_names=["CLASS_A", "CLASS_B"])
        image = np.zeros((1, 150, 150, 3))
        
        predictions = inference.predict(image)
        
        assert predictions["CLASS_A"] == 0.3
        assert predictions["CLASS_B"] == 0.7
    
    def test_get_prediction_label(self):
        """Test getting prediction label."""
        model = Mock()
        model.predict.return_value = np.array([[0.8]])
        
        inference = ModelInference(model, class_names=["NORMAL", "PNEUMONIA"])
        image = np.zeros((1, 150, 150, 3))
        
        label, confidence = inference.get_prediction_label(image)
        
        assert label == "PNEUMONIA"
        assert abs(confidence - 0.8) < 0.01


class TestGetPretrainedModel:
    """Tests for get_pretrained_model function."""
    
    @pytest.mark.slow
    def test_get_vgg19_model(self):
        """Test getting VGG19 model."""
        model = get_pretrained_model(
            model_name="vgg19",
            weights=None,  # Don't download ImageNet weights in tests
            input_shape=(150, 150, 3),
            num_classes=1,
        )
        
        assert model is not None
        assert model.input_shape == (None, 150, 150, 3)
        assert model.output_shape == (None, 1)
    
    @pytest.mark.slow
    def test_get_densenet201_model(self):
        """Test getting DenseNet201 model."""
        model = get_pretrained_model(
            model_name="densenet201",
            weights=None,
            input_shape=(150, 150, 3),
            num_classes=1,
        )
        
        assert model is not None
        assert model.input_shape == (None, 150, 150, 3)
        assert model.output_shape == (None, 1)
    
    def test_invalid_model_name(self):
        """Test with invalid model name."""
        with pytest.raises(ValueError, match="Unknown model name"):
            get_pretrained_model(
                model_name="invalid_model",
                weights=None,
                input_shape=(150, 150, 3),
                num_classes=1,
            )
    
    @pytest.mark.slow
    def test_multiclass_output(self):
        """Test creating model with multiple classes."""
        model = get_pretrained_model(
            model_name="vgg19",
            weights=None,
            input_shape=(150, 150, 3),
            num_classes=3,
        )
        
        assert model.output_shape == (None, 3)
