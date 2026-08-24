"""Prediction module for the Bank Customer Churn Prediction project.

This module provides inference functions for both:
- already transformed data
- raw customer data

For production inference, raw data should go through the exact same
feature engineering, preprocessing, and feature selection steps used
during training.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.feature_engineering import engineer_features


def predict(
    model: Any,
    X: pd.DataFrame,
) -> np.ndarray:
    """Generate class predictions from already transformed data.

    Parameters
    ----------
    model : Any
        Trained XGBoost model.

    X : pd.DataFrame
        Data already processed and feature-selected.

    Returns
    -------
    np.ndarray
        Predicted class labels.
    """

    return model.predict(X)


def predict_proba(
    model: Any,
    X: pd.DataFrame,
) -> np.ndarray:
    """Generate churn probabilities from already transformed data.

    Parameters
    ----------
    model : Any
        Trained XGBoost model.

    X : pd.DataFrame
        Data already processed and feature-selected.

    Returns
    -------
    np.ndarray
        Probability of churn.
    """

    return model.predict_proba(X)[:, 1]


def prepare_for_prediction(
    X: pd.DataFrame,
    preprocessor: Any,
    selector: Any,
) -> pd.DataFrame:
    """Transform raw data for model inference.

    Pipeline:

        raw data
            ↓
        feature engineering
            ↓
        preprocessing
            ↓
        feature selection

    No fitting is performed during inference.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if X.empty:
        raise ValueError("X is empty.")

    # --------------------------------------------------------------
    # 1. Feature engineering
    # --------------------------------------------------------------

    X_engineered = engineer_features(X)

    # --------------------------------------------------------------
    # 2. Preprocessing
    # --------------------------------------------------------------

    X_processed_array = preprocessor.transform(
        X_engineered
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    X_processed = pd.DataFrame(
        X_processed_array,
        columns=feature_names,
        index=X.index,
    )

    # --------------------------------------------------------------
    # 3. Verify selector compatibility
    # --------------------------------------------------------------

    if hasattr(selector, "feature_names_in_"):

        expected_features = list(
            selector.feature_names_in_
        )

        actual_features = list(
            X_processed.columns
        )

        if expected_features != actual_features:
            raise ValueError(
                "Preprocessor and selector are incompatible.\n"
                f"Selector expects {len(expected_features)} features.\n"
                f"Preprocessor produced {len(actual_features)} features.\n"
                f"Missing features: "
                f"{sorted(set(expected_features) - set(actual_features))}\n"
                f"Unexpected features: "
                f"{sorted(set(actual_features) - set(expected_features))}"
            )

    # --------------------------------------------------------------
    # 4. Feature selection
    # --------------------------------------------------------------

    X_selected_array = selector.transform(
        X_processed
    )

    selected_feature_names = (
        selector.get_feature_names_out(
            X_processed.columns
        )
    )

    X_selected = pd.DataFrame(
        X_selected_array,
        columns=selected_feature_names,
        index=X.index,
    )

    return X_selected


def predict_raw(
    model: Any,
    preprocessor: Any,
    selector: Any,
    X: pd.DataFrame,
) -> np.ndarray:
    """Generate predictions directly from raw customer data.

    Pipeline:

        raw data
            ↓
        feature engineering
            ↓
        preprocessing
            ↓
        feature selection
            ↓
        model
            ↓
        prediction

    Parameters
    ----------
    model : Any
        Trained XGBoost model.

    preprocessor : Any
        Fitted preprocessing transformer.

    selector : Any
        Fitted feature selector.

    X : pd.DataFrame
        Raw customer data.

    Returns
    -------
    np.ndarray
        Predicted class labels.
    """

    X_ready = prepare_for_prediction(
        X=X,
        preprocessor=preprocessor,
        selector=selector,
    )

    return predict(
        model=model,
        X=X_ready,
    )


def predict_raw_proba(
    model: Any,
    preprocessor: Any,
    selector: Any,
    X: pd.DataFrame,
) -> np.ndarray:
    """Generate churn probabilities directly from raw customer data.

    Parameters
    ----------
    model : Any
        Trained XGBoost model.

    preprocessor : Any
        Fitted preprocessing transformer.

    selector : Any
        Fitted feature selector.

    X : pd.DataFrame
        Raw customer data.

    Returns
    -------
    np.ndarray
        Churn probabilities.
    """

    X_ready = prepare_for_prediction(
        X=X,
        preprocessor=preprocessor,
        selector=selector,
    )

    return predict_proba(
        model=model,
        X=X_ready,
    )
