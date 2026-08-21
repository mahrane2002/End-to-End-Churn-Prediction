import pandas as pd

from src.data.data_ingestion import load_data


def test_load_data():
    df = load_data()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Churn" in df.columns