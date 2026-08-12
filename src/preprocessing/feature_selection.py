"""Feature selection module for the Bank Customer Churn Prediction project.

Responsibilities:
- Remove non-predictive identifier columns if present.
- Select top features using ANOVA F-value.
- Fit the selector only on the provided training data.
"""

from typing import Union

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif


IDENTIFIER_COLUMNS = {
    "RowNumber",
    "CustomerId",
    "Surname",
    "numerical__RowNumber",
    "numerical__CustomerId",
}


def remove_identifier_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Remove non-predictive identifier columns if present."""

    cols_to_drop = [
        col
        for col in df.columns
        if (
            col in IDENTIFIER_COLUMNS
            or col.startswith("categorical__Surname_")
        )
    ]

    if cols_to_drop:
        return df.drop(columns=cols_to_drop)

    return df


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame | None = None,
    k: Union[int, str] = 15,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame | None,
    SelectKBest,
]:
    """Fit SelectKBest only on training data and transform datasets."""

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")

    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    if X_train.empty:
        raise ValueError("X_train is empty.")

    if y_train.empty:
        raise ValueError("y_train is empty.")

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train must contain the same number of samples."
        )

    # Remove identifiers consistently
    X_train_clean = remove_identifier_columns(X_train)

    X_test_clean = (
        remove_identifier_columns(X_test)
        if X_test is not None
        else None
    )

    if X_train_clean.shape[1] == 0:
        raise ValueError(
            "No features remain after removing identifier columns."
        )

    # Prevent k from being larger than the number of available features
    if isinstance(k, int):
        k_actual = min(k, X_train_clean.shape[1])
    else:
        k_actual = k

    # IMPORTANT:
    # The selector is fitted ONLY on X_train.
    selector = SelectKBest(
        score_func=f_classif,
        k=k_actual,
    )

    selector.fit(
        X_train_clean,
        y_train,
    )

    selected_feature_names = selector.get_feature_names_out(
        X_train_clean.columns
    )

    # Transform training data
    X_train_selected = pd.DataFrame(
        selector.transform(X_train_clean),
        columns=selected_feature_names,
        index=X_train_clean.index,
    )

    # Transform optional second dataset
    X_test_selected = None

    if X_test_clean is not None:
        X_test_selected = pd.DataFrame(
            selector.transform(X_test_clean),
            columns=selected_feature_names,
            index=X_test_clean.index,
        )

    return (
        X_train_selected,
        X_test_selected,
        selector,
    )