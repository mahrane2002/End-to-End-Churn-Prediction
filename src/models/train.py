"""Model training module for the Bank Customer Churn Prediction project."""

from typing import Any

import pandas as pd

from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier

from src.config.config import RANDOM_STATE


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict[str, Any],
) -> XGBClassifier:
    """
    Train the final XGBoost churn prediction model.

    SMOTETomek is applied only to the training data.
    The XGBoost hyperparameters are provided by the
    Optuna tuning step.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.

    y_train : pd.Series
        Training target.

    best_params : dict[str, Any]
        Best hyperparameters found by Optuna.

    Returns
    -------
    XGBClassifier
        Trained final XGBoost model.
    """

    # ------------------------------------------------------------------
    # 1. Handle class imbalance
    # ------------------------------------------------------------------

    sampler = SMOTETomek(
        random_state=RANDOM_STATE,
    )

    X_train_resampled, y_train_resampled = sampler.fit_resample(
        X_train,
        y_train,
    )

    # ------------------------------------------------------------------
    # 2. Create final XGBoost model
    # ------------------------------------------------------------------

    model = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        enable_categorical=False,
    )

    # ------------------------------------------------------------------
    # 3. Train final model
    # ------------------------------------------------------------------

    model.fit(
        X_train_resampled,
        y_train_resampled,
    )

    return model