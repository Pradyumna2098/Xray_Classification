"""Model loading and inference utilities."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

logger = logging.getLogger(__name__)


class ModelInference:
    """Wrapper class for model inference operations."""
    
    def __init__(self, model: Model, class_names: list[str] | None = None):
        """Initialize the inference wrapper.
        
        Args:
            model: Loaded Keras model
            class_names: List of class names in order
        """
        self.model = model
        self.class_names = class_names or ["NORMAL", "PNEUMONIA"]
    
    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """Make prediction on a preprocessed image.
        
        Args:
            image: Preprocessed image array with shape (1, height, width, channels)
            
        Returns:
            Dictionary mapping class names to probabilities
        """
        predictions = self.model.predict(image, verbose=0)
        
        # Handle binary classification
        if predictions.shape[-1] == 1:
            # Sigmoid output
            prob_positive = float(predictions[0][0])
            prob_negative = 1.0 - prob_positive
            return {
                self.class_names[0]: prob_negative,
                self.class_names[1]: prob_positive,
            }
        else:
            # Softmax output
            probs = predictions[0]
            return {
                name: float(prob) 
                for name, prob in zip(self.class_names, probs)
            }
    
    def get_prediction_label(self, image: np.ndarray) -> Tuple[str, float]:
        """Get the predicted class label and confidence.
        
        Args:
            image: Preprocessed image array
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        predictions = self.predict(image)
        predicted_class = max(predictions, key=predictions.get)
        confidence = predictions[predicted_class]
        return predicted_class, confidence


def load_model_from_weights(
    model_architecture: Model,
    weights_path: str | Path,
) -> Model:
    """Load a model with pre-trained weights.
    
    Args:
        model_architecture: Compiled model architecture
        weights_path: Path to the .h5 weights file
        
    Returns:
        Model with loaded weights
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    model_architecture.load_weights(str(weights_path))
    logger.info(f"Loaded weights from {weights_path}")
    return model_architecture


def get_pretrained_model(
    model_name: str,
    weights: str = "imagenet",
    input_shape: Tuple[int, int, int] = (150, 150, 3),
    num_classes: int = 1,
) -> Model:
    """Get a pretrained model architecture.
    
    Args:
        model_name: Name of the model ('vgg19' or 'densenet201')
        weights: Initial weights ('imagenet' or None)
        input_shape: Input shape for the model
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    from tensorflow.keras.applications import DenseNet201, VGG19
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.models import Model as KerasModel
    
    # Load base model
    if model_name.lower() == "vgg19":
        base_model = VGG19(
            weights=weights,
            include_top=False,
            input_shape=input_shape,
        )
    elif model_name.lower() == "densenet201":
        base_model = DenseNet201(
            weights=weights,
            include_top=False,
            input_shape=input_shape,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    if num_classes == 1:
        output = Dense(1, activation='sigmoid', name='output')(x)
    else:
        output = Dense(num_classes, activation='softmax', name='output')(x)
    
    model = KerasModel(inputs=base_model.input, outputs=output)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy',
        metrics=['accuracy'],
    )
    
    return model
