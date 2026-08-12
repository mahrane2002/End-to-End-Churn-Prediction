"""Feature engineering module for the Bank Customer Churn Prediction project."""

import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create meaningful features for churn prediction.

    The function only creates new features.
    Data cleaning, imputation, scaling, and encoding
    are handled separately in preprocessing.py.

    Parameters
    ----------
    df : pd.DataFrame
        Input feature dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with engineered features.
    """

    data = df.copy()

    # ------------------------------------------------------------------
    # 1. Balance relative to estimated age
    # ------------------------------------------------------------------

    data["BalancePerAge"] = (
        data["Balance"]
        / data["Age"].replace(0, 1)
    )

    # ------------------------------------------------------------------
    # 2. Estimated customer activity
    # ------------------------------------------------------------------

    data["ProductsPerTenure"] = (
        data["NumOfProducts"]
        / data["Tenure"].replace(0, 1)
    )

    # ------------------------------------------------------------------
    # 3. Customer engagement indicator
    # ------------------------------------------------------------------

    data["IsActiveAndHasCard"] = (
        data["IsActiveMember"]
        * data["HasCrCard"]
    )

    # ------------------------------------------------------------------
    # 4. Age groups
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
    # 5. Balance status
    # ------------------------------------------------------------------

    data["HasBalance"] = (
        data["Balance"] > 0
    ).astype(int)

    # ------------------------------------------------------------------
    # 6. Product usage indicator
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
        Dataset with engineered features.
    """

    # ------------------------------------------------------------------
    # 1. Validate input type
    # ------------------------------------------------------------------

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input data must be a pandas DataFrame."
        )

    # ------------------------------------------------------------------
    # 2. Validate that the DataFrame is not empty
    # ------------------------------------------------------------------

    if df.empty:
        raise ValueError(
            "Input DataFrame is empty."
        )

    # ------------------------------------------------------------------
    # 3. Create engineered features
    # ------------------------------------------------------------------

    return create_features(df)