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

    The function applies:

        raw data
            ↓
        feature engineering
            ↓
        preprocessing
            ↓
        feature selection

    IMPORTANT
    ---------
    Neither the preprocessor nor the selector is fitted here.

    They must already be fitted on training data.

    Parameters
    ----------
    X : pd.DataFrame
        Raw customer features.

    preprocessor : Any
        Fitted preprocessing transformer.

    selector : Any
        Fitted feature selector.

    Returns
    -------
    pd.DataFrame
        Data ready to be passed to the trained model.
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
    #
    # IMPORTANT:
    # transform() only.
    # We NEVER call fit() or fit_transform() here.
    # --------------------------------------------------------------

    X_processed = preprocessor.transform(X_engineered)

    # --------------------------------------------------------------
    # 3. Recover feature names
    # --------------------------------------------------------------

    feature_names = preprocessor.get_feature_names_out()

    X_processed = pd.DataFrame(
        X_processed,
        columns=feature_names,
        index=X.index,
    )

    # --------------------------------------------------------------
    # 4. Feature selection
    #
    # IMPORTANT:
    # selector was fitted during training.
    # Only transform new data.
    # --------------------------------------------------------------

    X_selected = selector.transform(X_processed)

    selected_feature_names = selector.get_feature_names_out(
        X_processed.columns
    )

    X_selected = pd.DataFrame(
        X_selected,
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


def predict_raw_with_threshold(
    model: Any,
    preprocessor: Any,
    selector: Any,
    X: pd.DataFrame,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions from raw data using a custom threshold.

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

    threshold : float, default=0.5
        Probability threshold used to classify churn.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Churn probabilities and predicted labels.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0.0 and 1.0."
        )

    probabilities = predict_raw_proba(
        model=model,
        preprocessor=preprocessor,
        selector=selector,
        X=X,
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return probabilities, predictions