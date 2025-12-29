"""Image preprocessing utilities for X-ray classification."""
from __future__ import annotations

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array


def preprocess_image(
    image: Image.Image | np.ndarray | str,
    target_size: tuple[int, int] = (150, 150),
    normalize: bool = True,
) -> np.ndarray:
    """Preprocess an X-ray image for model inference.
    
    Args:
        image: Input image as PIL Image, numpy array, or file path
        target_size: Target size for resizing (height, width)
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        Preprocessed image array with shape (1, height, width, channels)
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = Image.open(image)
    
    # Convert to PIL Image if numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Ensure RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize(target_size)
    
    # Convert to array
    img_array = img_to_array(image)
    
    # Normalize if requested
    if normalize:
        img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def validate_image(image: Image.Image | np.ndarray) -> bool:
    """Validate that the image is suitable for processing.
    
    Args:
        image: Input image to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if isinstance(image, np.ndarray):
            # Check dimensions
            if len(image.shape) not in [2, 3]:
                return False
            # Check if grayscale or RGB
            if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
                return False
        elif isinstance(image, Image.Image):
            # PIL Image is valid if it can be converted to RGB
            return True
        else:
            return False
        return True
    except Exception:
        return False
