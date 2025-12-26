"""Grad-CAM implementation for model explainability."""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

logger = logging.getLogger(__name__)


class GradCAM:
    """Gradient-weighted Class Activation Mapping for model explainability."""
    
    def __init__(
        self,
        model: Model,
        layer_name: Optional[str] = None,
    ):
        """Initialize Grad-CAM.
        
        Args:
            model: Trained Keras model
            layer_name: Name of the convolutional layer to use for Grad-CAM.
                       If None, uses the last convolutional layer.
        """
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()
        
        if not self.layer_name:
            raise ValueError("Could not find a convolutional layer in the model")
        
        logger.info(f"Using layer '{self.layer_name}' for Grad-CAM")
    
    def _find_last_conv_layer(self) -> Optional[str]:
        """Find the last convolutional layer in the model.
        
        Returns:
            Name of the last convolutional layer or None
        """
        for layer in reversed(self.model.layers):
            # Check if it's a Conv2D layer
            if len(layer.output_shape) == 4:
                return layer.name
        return None
    
    def generate_heatmap(
        self,
        image: np.ndarray,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap.
        
        Args:
            image: Preprocessed image array (1, H, W, C)
            class_idx: Index of the class to visualize. If None, uses predicted class.
            
        Returns:
            Heatmap as numpy array (H, W)
        """
        # Create a model that maps the input to the activations of the target layer
        # and the output predictions
        grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output,
            ],
        )
        
        # Compute the gradient of the top predicted class for the input image
        # with respect to the activations of the target layer
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            
            # If class_idx is None, use the predicted class
            if class_idx is None:
                if predictions.shape[-1] == 1:
                    # Binary classification with sigmoid
                    class_idx = 0
                else:
                    # Multi-class classification
                    class_idx = tf.argmax(predictions[0])
            
            # Get the score for the target class
            if predictions.shape[-1] == 1:
                # Binary classification
                class_score = predictions[:, 0]
            else:
                # Multi-class classification
                class_score = predictions[:, class_idx]
        
        # Compute gradients
        grads = tape.gradient(class_score, conv_outputs)
        
        # Compute the guided gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by the computed gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        return heatmap
    
    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        original_image: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Overlay heatmap on original image.
        
        Args:
            heatmap: Grad-CAM heatmap (H, W)
            original_image: Original image (H, W, C) or (1, H, W, C)
            alpha: Transparency of the heatmap overlay
            colormap: OpenCV colormap to use
            
        Returns:
            Image with overlaid heatmap
        """
        # Remove batch dimension if present
        if len(original_image.shape) == 4:
            original_image = original_image[0]
        
        # Get dimensions
        height, width = original_image.shape[:2]
        
        # Resize heatmap to match image dimensions
        heatmap = cv2.resize(heatmap, (width, height))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        
        # Convert original image to uint8 if needed
        if original_image.max() <= 1.0:
            original_image = np.uint8(255 * original_image)
        else:
            original_image = np.uint8(original_image)
        
        # Ensure original image is RGB
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        elif original_image.shape[2] == 1:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        # Overlay heatmap on original image
        overlaid = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        
        return overlaid
    
    def explain_prediction(
        self,
        image: np.ndarray,
        class_idx: Optional[int] = None,
        alpha: float = 0.4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Grad-CAM explanation for a prediction.
        
        Args:
            image: Preprocessed image array (1, H, W, C)
            class_idx: Class index to explain. If None, explains predicted class.
            alpha: Transparency of the overlay
            
        Returns:
            Tuple of (heatmap, overlaid_image)
        """
        # Generate heatmap
        heatmap = self.generate_heatmap(image, class_idx)
        
        # Overlay on original image
        overlaid = self.overlay_heatmap(heatmap, image, alpha)
        
        return heatmap, overlaid


def get_gradcam_for_model(
    model: Model,
    model_type: str = "vgg19",
) -> GradCAM:
    """Get appropriate Grad-CAM instance for a model.
    
    Args:
        model: Trained Keras model
        model_type: Type of model ('vgg19', 'densenet201', or 'custom')
        
    Returns:
        Configured GradCAM instance
    """
    # Define last conv layers for known architectures
    last_conv_layers = {
        "vgg19": "block5_conv4",
        "densenet201": "conv5_block32_concat",
        "custom": None,  # Will auto-detect
    }
    
    layer_name = last_conv_layers.get(model_type.lower())
    
    return GradCAM(model, layer_name=layer_name)
