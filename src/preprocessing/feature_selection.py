"""Feature selection module for the Bank Customer Churn Prediction project.

Responsibilities:
- Remove non-predictive identifier columns if present.
- Select top features based on statistical score (ANOVA F-value).
- Ensure zero data leakage by fitting selector strictly on training data.
"""

from typing import Union

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

# List of non-predictive identifier columns to drop if present
IDENTIFIER_COLUMNS = {
    "RowNumber",
    "CustomerId",
    "Surname",
    "numerical__RowNumber",
    "numerical__CustomerId",
}


def remove_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove non-predictive identifier columns from DataFrame if present."""
    cols_to_drop = [
        col
        for col in df.columns
        if col in IDENTIFIER_COLUMNS or col.startswith("categorical__Surname_")
    ]
    if cols_to_drop:
        return df.drop(columns=cols_to_drop)
    return df


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame | None = None,
    k: Union[int, str] = 15,
) -> tuple[pd.DataFrame, pd.DataFrame | None, SelectKBest]:
    """Select top K features using SelectKBest with f_classif.

    The feature selector is fitted ONLY on the training data (X_train, y_train).
    X_test is transformed using the fitted selector to prevent data leakage.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature dataset.

    y_train : pd.Series
        Training target variable.

    X_test : pd.DataFrame or None, default=None
        Test feature dataset.

    k : int or str, default=15
        Number of top features to select. If k is larger than total features,
        all available features are selected. Can also be 'all'.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame | None, SelectKBest]
        - X_train_selected : Training features with selected columns.
        - X_test_selected  : Test features with selected columns (or None).
        - selector         : Fitted SelectKBest object.
    """

    # 1. Input validation
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

    # 2. Filter out non-predictive identifier columns if present
    X_train_clean = remove_identifier_columns(X_train)
    X_test_clean = (
        remove_identifier_columns(X_test) if X_test is not None else None
    )

    # 3. Determine actual K to prevent ValueError if features < k
    total_features = X_train_clean.shape[1]
    if isinstance(k, int):
        k_actual = min(k, total_features)
    else:
        k_actual = k

    # 4. Instantiate and fit selector ONLY on training data
    selector = SelectKBest(score_func=f_classif, k=k_actual)
    selector.fit(X_train_clean, y_train)

    # 5. Transform X_train
    X_train_transformed = selector.transform(X_train_clean)
    selected_feature_names = selector.get_feature_names_out(
        X_train_clean.columns
    )

    X_train_selected = pd.DataFrame(
        X_train_transformed,
        columns=selected_feature_names,
        index=X_train_clean.index,
    )

    # 6. Transform X_test if provided
    X_test_selected = None
    if X_test_clean is not None:
        X_test_transformed = selector.transform(X_test_clean)
        X_test_selected = pd.DataFrame(
            X_test_transformed,
            columns=selected_feature_names,
            index=X_test_clean.index,
        )

    return X_train_selected, X_test_selected, selector
