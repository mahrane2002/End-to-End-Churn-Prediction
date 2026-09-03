"""
config.py
---------
Centralized configuration for the Bank Customer Churn Prediction project.

Responsibilities:
    - Define all global project paths (using pathlib.Path).
    - Define dataset-related constants.
    - Define ML training parameters and model constants.

Usage:
    from src.config.config import DATA_DIR, TARGET_COLUMN, RANDOM_STATE
"""

from pathlib import Path

# ==============================================================================
# PROJECT ROOT
# ==============================================================================
# Resolves to the project root directory (two levels up from this file):
# src/config/config.py -> src/config -> src -> project root
ROOT_DIR: Path = Path(__file__).resolve().parents[2]


# ==============================================================================
# DATA PATHS
# ==============================================================================

# Top-level data directory
DATA_DIR: Path = ROOT_DIR / "data"

# Raw data — source files, never modified
RAW_DATA_DIR: Path = DATA_DIR / "raw"
RAW_DATA_PATH: Path = RAW_DATA_DIR / "Churn_Modelling.csv"

# Processed data — output of preprocessing pipeline
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"


# ==============================================================================
# ARTIFACT PATHS
# ==============================================================================
# Serialized objects produced during training (models, transformers, metadata)

ARTIFACT_DIR: Path = ROOT_DIR / "artifacts"

MODEL_PATH: Path = ARTIFACT_DIR / "model.pkl"
PREPROCESSOR_PATH: Path = ARTIFACT_DIR / "preprocessor.pkl"
THRESHOLD_PATH: Path = ARTIFACT_DIR / "threshold.json"
SELECTOR_PATH: Path = ARTIFACT_DIR / "selector.pkl"
METADATA_PATH: Path = ARTIFACT_DIR / "metadata.json"
INFERENCE_PIPELINE_PATH = ARTIFACT_DIR / "inference_pipeline.pkl"


# ==============================================================================
# DATASET CONSTANTS
# ==============================================================================

# Name of the target column as it appears in the raw CSV
ORIGINAL_TARGET_COLUMN: str = "Exited"

# Name of the target column used internally throughout the pipeline
TARGET_COLUMN: str = "Churn"


# ==============================================================================
# ML GLOBAL PARAMETERS
# ==============================================================================

# Seed used across all random operations (train/test split, models, etc.)
RANDOM_STATE: int = 42

# Fraction of the dataset reserved for the test set
TEST_SIZE: float = 0.2


# ==============================================================================
# TRAINING PARAMETERS
# ==============================================================================

# Number of folds used in cross-validation
CV_FOLDS: int = 5

# Metric optimized during model selection and cross-validation
SCORING_METRIC: str = "roc_auc"


# ==============================================================================
# MODEL CONSTANTS
# ==============================================================================

# Logical name of the best model selected after training
MODEL_NAME: str = "best_model"

SHAP_DIR = ARTIFACT_DIR / "shap"
SHAP_BACKGROUND_PATH = SHAP_DIR / "background.pkl"