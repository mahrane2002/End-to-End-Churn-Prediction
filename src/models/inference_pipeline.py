"""Production inference pipeline for customer churn prediction."""

from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.feature_engineering import engineer_features


class ChurnInferencePipeline:
    """
    Complete inference pipeline for raw customer data.

    Pipeline:

        raw customer data
            ↓
        feature engineering
            ↓
        preprocessing
            ↓
        feature selection
            ↓
        XGBoost model
            ↓
        prediction
    """

    def __init__(
        self,
        model: Any,
        preprocessor: Any,
        selector: Any,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.selector = selector

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform raw customer data into model-ready features.
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a pandas DataFrame."
            )

        if X.empty:
            raise ValueError(
                "X is empty."
            )

        # ----------------------------------------------------------
        # 1. Feature engineering
        # ----------------------------------------------------------

        X_engineered = engineer_features(X)

        # ----------------------------------------------------------
        # 2. Preprocessing
        # ----------------------------------------------------------

        X_processed_array = (
            self.preprocessor.transform(
                X_engineered
            )
        )

        feature_names = (
            self.preprocessor
            .get_feature_names_out()
        )

        X_processed = pd.DataFrame(
            X_processed_array,
            columns=feature_names,
            index=X.index,
        )

        # ----------------------------------------------------------
        # 3. Verify selector compatibility
        # ----------------------------------------------------------

        if hasattr(
            self.selector,
            "feature_names_in_",
        ):

            expected_features = list(
                self.selector.feature_names_in_
            )

            actual_features = list(
                X_processed.columns
            )

            if expected_features != actual_features:
                raise ValueError(
                    "Preprocessor and selector "
                    "are incompatible."
                )

        # ----------------------------------------------------------
        # 4. Feature selection
        # ----------------------------------------------------------

        X_selected_array = (
            self.selector.transform(
                X_processed
            )
        )

        selected_feature_names = (
            self.selector.get_feature_names_out(
                X_processed.columns
            )
        )

        X_selected = pd.DataFrame(
            X_selected_array,
            columns=selected_feature_names,
            index=X.index,
        )

        return X_selected

    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Generate class predictions from raw data.
        """

        X_ready = self.transform(X)

        return self.model.predict(
            X_ready
        )

    def predict_proba(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Generate churn probabilities from raw data.
        """

        X_ready = self.transform(X)

        return self.model.predict_proba(
            X_ready
        )[:, 1]