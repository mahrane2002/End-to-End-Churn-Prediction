"""Feature engineering module for the Bank Customer Churn Prediction project."""

import pandas as pd


# ---------------------------------------------------------------------------
# Columns that must never be used as model features
# ---------------------------------------------------------------------------

IDENTIFIER_COLUMNS = [
    "RowNumber",
    "CustomerId",
    "Surname",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create meaningful features for churn prediction.

    Identifier columns are removed before preprocessing so that they
    are never one-hot encoded or passed to feature selection.

    Data cleaning, imputation, scaling, and encoding
    are handled separately in preprocessing.py.

    Parameters
    ----------
    df : pd.DataFrame
        Input feature dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with engineered features and without identifiers.
    """

    data = df.copy()

    # ------------------------------------------------------------------
    # 1. Remove identifiers
    # ------------------------------------------------------------------

    columns_to_drop = [
        column
        for column in IDENTIFIER_COLUMNS
        if column in data.columns
    ]

    if columns_to_drop:
        data = data.drop(columns=columns_to_drop)

    # ------------------------------------------------------------------
    # 2. Balance relative to estimated age
    # ------------------------------------------------------------------

    data["BalancePerAge"] = (
        data["Balance"]
        / data["Age"].replace(0, 1)
    )

    # ------------------------------------------------------------------
    # 3. Estimated customer activity
    # ------------------------------------------------------------------

    data["ProductsPerTenure"] = (
        data["NumOfProducts"]
        / data["Tenure"].replace(0, 1)
    )

    # ------------------------------------------------------------------
    # 4. Customer engagement indicator
    # ------------------------------------------------------------------

    data["IsActiveAndHasCard"] = (
        data["IsActiveMember"]
        * data["HasCrCard"]
    )

    # ------------------------------------------------------------------
    # 5. Age groups
    # ------------------------------------------------------------------

    data["AgeGroup"] = pd.cut(
        data["Age"],
        bins=[0, 30, 40, 50, 60, float("inf")],
        labels=[
            "Young",
            "Adult",
            "Middle_Aged",
            "Senior",
            "Older",
        ],
        right=False,
    )

    # ------------------------------------------------------------------
    # 6. Balance status
    # ------------------------------------------------------------------

    data["HasBalance"] = (
        data["Balance"] > 0
    ).astype(int)

    # ------------------------------------------------------------------
    # 7. Product usage indicator
    # ------------------------------------------------------------------

    data["MultipleProducts"] = (
        data["NumOfProducts"] > 1
    ).astype(int)

    return data


def engineer_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the input and create engineered features.

    Parameters
    ----------
    df : pd.DataFrame
        Input feature dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with engineered features and without identifiers.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input data must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Input DataFrame is empty."
        )

    return create_features(df)