"""Artifact manager module for saving and loading machine learning artifacts."""

import json
from pathlib import Path
from typing import Any
import joblib

from src.config.config import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    SELECTOR_PATH,
    METADATA_PATH,
    INFERENCE_PIPELINE_PATH
)


def save_model(model: Any, path: Path = MODEL_PATH) -> None:
    """Save a trained model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: Path = MODEL_PATH) -> Any:
    """Load a trained model from disk."""
    if not path.exists():
        raise FileNotFoundError(f"No model found at {path}")
    return joblib.load(path)


def save_preprocessor(preprocessor: Any, path: Path = PREPROCESSOR_PATH) -> None:
    """Save a fitted preprocessor to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"Preprocessor saved to {path}")


def load_preprocessor(path: Path = PREPROCESSOR_PATH) -> Any:
    """Load a fitted preprocessor from disk."""
    if not path.exists():
        raise FileNotFoundError(f"No preprocessor found at {path}")
    return joblib.load(path)


def save_selector(selector: Any, path: Path = SELECTOR_PATH) -> None:
    """Save a fitted feature selector to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selector, path)
    print(f"Selector saved to {path}")


def load_selector(path: Path = SELECTOR_PATH) -> Any:
    """Load a fitted feature selector from disk."""
    if not path.exists():
        raise FileNotFoundError(f"No selector found at {path}")
    return joblib.load(path)


def save_metadata(metadata: dict[str, Any], path: Path = METADATA_PATH) -> None:
    """Save pipeline metadata as a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"Metadata saved to {path}")


def load_metadata(path: Path = METADATA_PATH) -> dict[str, Any]:
    """Load pipeline metadata from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"No metadata found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_inference_pipeline(
    pipeline: Any,
    path: Path = INFERENCE_PIPELINE_PATH,
) -> None:
    """Save the complete inference pipeline."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        path,
    )

    print(
        f"Inference pipeline saved to {path}"
    )


def load_inference_pipeline(
    path: Path = INFERENCE_PIPELINE_PATH,
) -> Any:
    """Load the complete inference pipeline."""

    if not path.exists():
        raise FileNotFoundError(
            f"No inference pipeline found at {path}"
        )

    return joblib.load(path)