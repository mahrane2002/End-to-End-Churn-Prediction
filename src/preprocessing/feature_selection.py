"""Simple feature selection module."""

import pandas as pd
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier

from src.config.config import RANDOM_STATE


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, SelectFromModel]:
    """Select important features using XGBoost.

    The selector is fitted only on the training data.
    """

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )

    selector = SelectFromModel(
        estimator=model,
        threshold="median",
    )

    X_train_selected = selector.fit_transform(
        X_train,
        y_train,
    )

    X_test_selected = selector.transform(X_test)

    selected_features = X_train.columns[
        selector.get_support()
    ]

    X_train_selected = pd.DataFrame(
        X_train_selected,
        columns=selected_features,
        index=X_train.index,
    )

    X_test_selected = pd.DataFrame(
        X_test_selected,
        columns=selected_features,
        index=X_test.index,
    )

    return (
        X_train_selected,
        X_test_selected,
        selector,
    )