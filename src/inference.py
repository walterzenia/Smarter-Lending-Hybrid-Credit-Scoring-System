"""Simple inference helper utilities (small stubs).

These are lightweight helpers to load a saved pipeline and run predictions.
"""
import joblib
from typing import Any


def load_pipeline(path: str):
    """Load a joblib pipeline from disk."""
    return joblib.load(path)


def predict(pipeline: Any, X):
    """Return predictions from a fitted pipeline."""
    return pipeline.predict(X)
