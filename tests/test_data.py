import pandas as pd
from src.config.config import RAW_DATA_PATH, TARGET_COLUMN
from src.data.data_ingestion import load_data


def test_load_data_returns_dataframe():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_load_data_target_column_renamed():
    df = load_data()
    assert TARGET_COLUMN in df.columns
    assert "Exited" not in df.columns


def test_load_data_custom_path(tmp_path):
    custom_file = tmp_path / "sample.csv"
    sample_df = pd.DataFrame({"CustomerId": [1, 2], "Exited": [0, 1]})
    sample_df.to_csv(custom_file, index=False)

    df = load_data(custom_file)
    assert len(df) == 2
    assert "Churn" in df.columns
