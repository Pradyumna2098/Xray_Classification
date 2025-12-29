"""Unit tests for preprocessing utilities."""
import numpy as np
import pytest
from PIL import Image

from src.utils.preprocessing import preprocess_image, validate_image


class TestPreprocessImage:
    """Tests for preprocess_image function."""
    
    def test_preprocess_pil_image(self):
        """Test preprocessing a PIL Image."""
        # Create a test image
        image = Image.new("RGB", (200, 200), color="red")
        
        # Preprocess
        result = preprocess_image(image, target_size=(150, 150), normalize=True)
        
        # Check shape
        assert result.shape == (1, 150, 150, 3)
        
        # Check normalization
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_preprocess_numpy_array(self):
        """Test preprocessing a numpy array."""
        # Create a test array
        array = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Preprocess
        result = preprocess_image(array, target_size=(150, 150), normalize=True)
        
        # Check shape
        assert result.shape == (1, 150, 150, 3)
    
    def test_preprocess_grayscale_image(self):
        """Test preprocessing a grayscale image."""
        # Create a grayscale image
        image = Image.new("L", (200, 200), color=128)
        
        # Preprocess
        result = preprocess_image(image, target_size=(150, 150), normalize=True)
        
        # Should be converted to RGB
        assert result.shape == (1, 150, 150, 3)
    
    def test_preprocess_without_normalization(self):
        """Test preprocessing without normalization."""
        image = Image.new("RGB", (200, 200), color=(255, 0, 0))
        
        result = preprocess_image(image, target_size=(150, 150), normalize=False)
        
        # Values should not be normalized
        assert result.max() > 1.0
    
    def test_preprocess_custom_target_size(self):
        """Test preprocessing with custom target size."""
        image = Image.new("RGB", (200, 200), color="blue")
        
        result = preprocess_image(image, target_size=(224, 224), normalize=True)
        
        assert result.shape == (1, 224, 224, 3)


class TestValidateImage:
    """Tests for validate_image function."""
    
    def test_validate_pil_image(self):
        """Test validating a PIL Image."""
        image = Image.new("RGB", (100, 100))
        assert validate_image(image) is True
    
    def test_validate_numpy_array_rgb(self):
        """Test validating an RGB numpy array."""
        array = np.zeros((100, 100, 3), dtype=np.uint8)
        assert validate_image(array) is True
    
    def test_validate_numpy_array_grayscale(self):
        """Test validating a grayscale numpy array."""
        array = np.zeros((100, 100), dtype=np.uint8)
        assert validate_image(array) is True
    
    def test_validate_invalid_dimensions(self):
        """Test validating array with invalid dimensions."""
        array = np.zeros((100,), dtype=np.uint8)
        assert validate_image(array) is False
    
    def test_validate_invalid_channels(self):
        """Test validating array with invalid number of channels."""
        array = np.zeros((100, 100, 5), dtype=np.uint8)
        assert validate_image(array) is False
    
    def test_validate_invalid_type(self):
        """Test validating invalid type."""
        assert validate_image("not an image") is False
        assert validate_image(123) is False
        assert validate_image(None) is False
