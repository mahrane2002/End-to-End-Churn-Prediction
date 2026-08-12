"""Prediction module for the Bank Customer Churn Prediction project."""

from typing import Any

import numpy as np
import pandas as pd


def predict(
    model: Any,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Generate churn predictions using a trained model.

    Parameters
    ----------
    model : Any
        Trained XGBoost classification model.

    X : pd.DataFrame
        Input features already transformed using the same
        preprocessing and feature selection steps used during training.

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
    """
    Generate churn probabilities using a trained model.

    Parameters
    ----------
    model : Any
        Trained XGBoost classification model.

    X : pd.DataFrame
        Input features already transformed using the same
        preprocessing and feature selection steps used during training.

    Returns
    -------
    np.ndarray
        Probability of churn for each observation.
    """

    return model.predict_proba(X)[:, 1]