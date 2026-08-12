"""Data validation module for the churn prediction project."""

from pathlib import Path

import pandas as pd

from src.config.config import RAW_DATA_PATH, TARGET_COLUMN
from src.data.data_ingestion import load_data


# Expected columns after data ingestion.
EXPECTED_COLUMNS = [
    "RowNumber",
    "CustomerId",
    "Surname",
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Churn",
]


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the dataset before preprocessing.

    Checks:
    - Dataset is not empty.
    - Expected columns are present.
    - Target column exists.
    - Target contains only 0 and 1.
    - No duplicate rows.
    - Missing values are reported.

    Returns:
        True if all critical validation checks pass.
    """

    # Check that the dataset is not empty.
    if df.empty:
        raise ValueError("Validation failed: dataset is empty.")

    # Check that all expected columns are present.
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Validation failed: missing required columns: {missing_columns}")

    # Check that the target column exists.
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Validation failed: missing required columns: {{'{TARGET_COLUMN}'}}")

    # Check that the target contains only binary values.
    target_values = set(df[TARGET_COLUMN].dropna().unique())

    if not target_values.issubset({0, 1}):
        raise ValueError(f"Validation failed: invalid target values: {target_values}")

    # Check for duplicate rows.
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(f"Validation failed: {duplicate_count} duplicate rows found.")

    # Report missing values without modifying the data.
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        print("Warning: missing values detected:")
        print(missing_values)

    print("Data validation passed.")

    return True


