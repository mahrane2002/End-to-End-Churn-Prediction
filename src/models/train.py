"""Model training module for the Bank Customer Churn Prediction project."""

import pandas as pd

from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBClassifier:
    """
    Train the XGBoost churn prediction model.

    SMOTETomek is applied only to the training data
    to handle class imbalance without contaminating
    the test set.

    Parameters
    ----------
    X_train : pd.DataFrame
        Selected training features.

    y_train : pd.Series
        Training target.

    Returns
    -------
    XGBClassifier
        Trained XGBoost model.
    """

    # ------------------------------------------------------------------
    # 1. Handle class imbalance
    # ------------------------------------------------------------------

    sampler = SMOTETomek(
        random_state=42,
    )

    X_train_resampled, y_train_resampled = sampler.fit_resample(
        X_train,
        y_train,
    )

    # ------------------------------------------------------------------
    # 2. Define XGBoost model
    # ------------------------------------------------------------------

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )

    # ------------------------------------------------------------------
    # 3. Train model on resampled training data
    # ------------------------------------------------------------------

    model.fit(
        X_train_resampled,
        y_train_resampled,
    )

    return model