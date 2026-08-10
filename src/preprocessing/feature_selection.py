
"""Feature selection module using Recursive Feature Elimination with CV."""

import pandas as pd

from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, RFECV]:
    """
    Select the most relevant features using RFECV.

    RFECV is fitted only on the training data to avoid
    data leakage. The same selected features are then
    applied
    to the test data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Preprocessed training features.

    y_train : pd.Series
        Training target.

    X_test : pd.DataFrame
        Preprocessed test features.

    Returns
    -------
    tuple
        Selected training features,
        selected test features,
        fitted RFECV selector.
    """

    estimator = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    selector = RFECV(
        estimator=estimator,
        step=1,
        cv=5,
        scoring="roc_auc",
        min_features_to_select=5,
        n_jobs=-1,
    )

    # Fit RFECV only on the training data.
    selector.fit(X_train, y_train)

    # Apply the selected features to both datasets.
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    # Keep the original feature names.
    selected_features = X_train.columns[selector.support_]

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

    return X_train_selected, X_test_selected, selector
