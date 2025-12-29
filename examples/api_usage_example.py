#!/usr/bin/env python3
"""
Example script demonstrating API usage for X-Ray Classification.

This script shows how to:
1. Check API health
2. Make predictions on X-ray images
3. Generate Grad-CAM explanations
4. Retrieve Prometheus metrics
"""

import argparse
import json
import sys
from pathlib import Path

import requests
from PIL import Image


def check_health(api_url: str) -> dict:
    """Check API health status."""
    print("Checking API health...")
    response = requests.get(f"{api_url}/health")
    response.raise_for_status()
    health = response.json()
    print(f"✓ API Status: {health['status']}")
    print(f"  Model Loaded: {health['model_loaded']}")
    print(f"  Model Name: {health['model_name']}")
    return health


def predict(api_url: str, image_path: Path) -> dict:
    """Make a prediction on an X-ray image."""
    print(f"\nMaking prediction on {image_path.name}...")
    
    # Open and validate image
    try:
        image = Image.open(image_path)
        print(f"  Image size: {image.size}")
        print(f"  Image mode: {image.mode}")
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        sys.exit(1)
    
    # Send request
    with open(image_path, "rb") as f:
        files = {"file": (image_path.name, f, "image/jpeg")}
        response = requests.post(f"{api_url}/predict", files=files)
    
    response.raise_for_status()
    result = response.json()
    
    print("✓ Prediction Results:")
    print(f"  Predicted Class: {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Probabilities:")
    for class_name, prob in result['probabilities'].items():
        print(f"    {class_name}: {prob:.2%}")
    print(f"  Inference Time: {result['inference_time_ms']:.2f}ms")
    
    return result


def explain(api_url: str, image_path: Path, output_path: Path, alpha: float = 0.4) -> None:
    """Generate Grad-CAM explanation for an image."""
    print(f"\nGenerating Grad-CAM explanation...")
    
    with open(image_path, "rb") as f:
        files = {"file": (image_path.name, f, "image/jpeg")}
        params = {"alpha": alpha}
        response = requests.post(f"{api_url}/explain", files=files, params=params)
    
    response.raise_for_status()
    
    # Save explanation image
    with open(output_path, "wb") as f:
        f.write(response.content)
    
    print(f"✓ Explanation saved to: {output_path}")
    print(f"  Alpha (transparency): {alpha}")


def get_metrics(api_url: str) -> str:
    """Retrieve Prometheus metrics."""
    print("\nRetrieving Prometheus metrics...")
    response = requests.get(f"{api_url}/metrics")
    response.raise_for_status()
    metrics = response.text
    
    # Parse and display key metrics
    lines = metrics.split("\n")
    key_metrics = [line for line in lines if not line.startswith("#") and line.strip()]
    
    print("✓ Sample Metrics:")
    for metric in key_metrics[:10]:  # Show first 10 metrics
        print(f"  {metric}")
    
    if len(key_metrics) > 10:
        print(f"  ... and {len(key_metrics) - 10} more metrics")
    
    return metrics


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Example script for X-Ray Classification API"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Path to X-ray image for prediction",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate Grad-CAM explanation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("explanation.png"),
        help="Output path for explanation image (default: explanation.png)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Transparency for Grad-CAM overlay (default: 0.4)",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Retrieve Prometheus metrics",
    )
    
    args = parser.parse_args()
    
    try:
        # Always check health first
        check_health(args.api_url)
        
        # Make prediction if image provided
        if args.image:
            if not args.image.exists():
                print(f"✗ Error: Image not found: {args.image}")
                sys.exit(1)
            
            predict(args.api_url, args.image)
            
            # Generate explanation if requested
            if args.explain:
                explain(args.api_url, args.image, args.output, args.alpha)
        
        # Retrieve metrics if requested
        if args.metrics:
            get_metrics(args.api_url)
        
        print("\n✓ All operations completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print(f"✗ Error: Could not connect to API at {args.api_url}")
        print("  Make sure the API is running (docker-compose up or uvicorn api.main:app)")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTP Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
