
"""Model training module for the Bank Customer Churn Prediction project."""

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data.data_ingestion import load_data
from src.data.data_validation import validate_data
from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.preprocessing import preprocess_data
from src.preprocessing.feature_selection import select_features


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LogisticRegression:
    """
    Train the churn prediction model.

    Parameters
    ----------
    X_train : pd.DataFrame
        Selected training features.

    y_train : pd.Series
        Training target.

    Returns
    -------
    LogisticRegression
        Trained logistic regression model.
    """

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model


def main() -> LogisticRegression:
    """
    Run the complete training pipeline.

    Returns
    -------
    LogisticRegression
        Trained model.
    """

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------

    df = load_data()

    # ------------------------------------------------------------------
    # 2. Validate data
    # ------------------------------------------------------------------

    df = validate_data(df)

    # ------------------------------------------------------------------
    # 3. Create engineered features
    # ------------------------------------------------------------------

    df = engineer_features(df)

    # ------------------------------------------------------------------
    # 4. Preprocess data
    # ------------------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    ) = preprocess_data(df)

    # ------------------------------------------------------------------
    # 5. Select relevant features using RFECV
    # ------------------------------------------------------------------

    (
        X_train_selected,
        X_test_selected,
        selector,
    ) = select_features(
        X_train,
        y_train,
        X_test,
    )

    # ------------------------------------------------------------------
    # 6. Train model
    # ------------------------------------------------------------------

    model = train_model(
        X_train_selected,
        y_train,
    )

    return model
