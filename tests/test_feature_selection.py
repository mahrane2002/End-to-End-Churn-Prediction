import pandas as pd

from src.preprocessing.feature_selection import (
    remove_identifier_columns,
    select_features,
)


def test_remove_identifier_columns():
    df = pd.DataFrame({
        "CustomerId": [1, 2],
        "Age": [30, 40],
        "Balance": [1000, 2000],
    })

    result = remove_identifier_columns(df)

    assert "CustomerId" not in result.columns
    assert "Age" in result.columns
    assert "Balance" in result.columns


def test_select_features():
    X = pd.DataFrame({
        "Age": [20, 30, 40, 50],
        "Balance": [1000, 2000, 3000, 4000],
        "CreditScore": [600, 650, 700, 750],
    })

    y = pd.Series([0, 0, 1, 1])

    X_selected, _, selector = select_features(
        X,
        y,
        k=2,
    )

    assert X_selected.shape[0] == X.shape[0]
    assert X_selected.shape[1] == 2
    assert selector is not None