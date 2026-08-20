import pytest
from src.data.data_validation import validate_data

def test_validate_data_success(sample_df):
    # Arrange: Use a valid dataframe

    # Act: Validate the data
    result = validate_data(sample_df)

    # Assert: Verification passes
    assert result is True

def test_validate_data_missing_column(sample_df):
    # Arrange: Remove a required column
    invalid_df = sample_df.drop(columns=["Age"])

    # Act & Assert: Verify it raises ValueError
    with pytest.raises(ValueError) as exc_info:
        validate_data(invalid_df)
    assert "missing required columns" in str(exc_info.value)

def test_validate_data_invalid_values(sample_df):
    # Arrange: Inject an invalid age (e.g., -5) and credit score outside [300, 850]
    invalid_df = sample_df.copy()
    invalid_df.loc[0, "Age"] = -5
    invalid_df.loc[1, "CreditScore"] = 250

    # Act & Assert: Verify it raises ValueError
    with pytest.raises(ValueError) as exc_info:
        validate_data(invalid_df)
    assert "Age <= 0" in str(exc_info.value) or "CreditScore outside" in str(exc_info.value)
