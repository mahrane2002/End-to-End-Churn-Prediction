import pandas as pd
from src.data.data_ingestion import load_data

def test_load_data_success(tmp_path, sample_raw_data):
    # Arrange: Save the raw sample data to a temporary CSV file
    temp_csv = tmp_path / "Churn_Modelling.csv"
    sample_raw_data.to_csv(temp_csv, index=False)

    # Act: Load the data from the temporary path
    df = load_data(temp_csv)

    # Assert: Verify the data was loaded properly
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Churn" in df.columns
    assert "Exited" not in df.columns
    assert df.shape[0] == len(sample_raw_data)
