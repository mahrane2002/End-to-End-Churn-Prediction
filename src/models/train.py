
"""Model training module for the Bank Customer Churn Prediction project."""

import pandas as pd
from xgboost import XGBClassifier

from src.data.data_ingestion import load_data
from src.data.data_validation import validate_data
from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.preprocessing import preprocess_data
from src.preprocessing.feature_selection import select_features


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBClassifier:
    """
    Train the XGBoost churn prediction model.

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

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(X_train, y_train)

    return model


def main() -> XGBClassifier:
    """Run the complete model training pipeline."""

    # 1. Load data
    df = load_data()

    # 2. Validate data
    df = validate_data(df)

    # 3. Create engineered features before the train/test split
    df = engineer_features(df)

    # 4. Split and preprocess the data
    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    ) = preprocess_data(df)

    # 5. Select relevant features using RFECV
    (
        X_train_selected,
        X_test_selected,
        selector,
    ) = select_features(
        X_train,
        y_train,
        X_test,
    )

    # 6. Train the final XGBoost model
    model = train_model(
        X_train_selected,
        y_train,
    )

    return model

