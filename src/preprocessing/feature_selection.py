"""Feature selection module for the Bank Customer Churn Prediction project.

Responsibilities:
- Select top features using ANOVA F-value.
- Fit the selector only on training data.
- Transform test/inference data using the fitted selector.
"""

from typing import Union

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif


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

    # --------------------------------------------------------------
    # Make sure no identifier reaches feature selection
    # --------------------------------------------------------------

    forbidden_columns = {
        "RowNumber",
        "CustomerId",
        "Surname",
    }

    forbidden_present = (
        forbidden_columns
        & set(X_train.columns)
    )

    if forbidden_present:
        raise ValueError(
            "Identifier columns must not reach feature selection: "
            f"{sorted(forbidden_present)}"
        )

    # --------------------------------------------------------------
    # Check test data
    # --------------------------------------------------------------

    if X_test is not None:
        if not isinstance(X_test, pd.DataFrame):
            raise TypeError(
                "X_test must be a pandas DataFrame."
            )

        forbidden_test = (
            forbidden_columns
            & set(X_test.columns)
        )

        if forbidden_test:
            raise ValueError(
                "Identifier columns must not reach feature selection: "
                f"{sorted(forbidden_test)}"
            )

    if X_train.shape[1] == 0:
        raise ValueError(
            "No features available for feature selection."
        )

    # --------------------------------------------------------------
    # Prevent k from being larger than available features
    # --------------------------------------------------------------

    if isinstance(k, int):
        k_actual = min(k, X_train.shape[1])
    else:
        k_actual = k

    # --------------------------------------------------------------
    # Fit selector ONLY on training data
    # --------------------------------------------------------------

    selector = SelectKBest(
        score_func=f_classif,
        k=k_actual,
    )

    selector.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------------
    # Selected feature names
    # --------------------------------------------------------------

    selected_feature_names = selector.get_feature_names_out(
        X_train.columns
    )

    # --------------------------------------------------------------
    # Transform training data
    # --------------------------------------------------------------

    X_train_selected = pd.DataFrame(
        selector.transform(X_train),
        columns=selected_feature_names,
        index=X_train.index,
    )

    # --------------------------------------------------------------
    # Transform optional test data
    # --------------------------------------------------------------

    X_test_selected = None

    if X_test is not None:
        X_test_selected = pd.DataFrame(
            selector.transform(X_test),
            columns=selected_feature_names,
            index=X_test.index,
        )

    return (
        X_train_selected,
        X_test_selected,
        selector,
    )