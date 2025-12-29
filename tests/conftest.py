"""Pytest configuration and fixtures."""
import os
import sys
from pathlib import Path

# Add src to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

# Skip model loading in all tests
os.environ["SKIP_MODEL_LOADING"] = "1"
