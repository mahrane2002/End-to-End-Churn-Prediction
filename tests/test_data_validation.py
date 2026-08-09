"""Tests for the data validation module."""

import pandas as pd
import pytest

from src.config.config import TARGET_COLUMN
from src.data.data_ingestion import load_data
from src.data.data_validation import validate_data


def test_validate_data_success():
    """Test that a valid DataFrame loaded via ingestion passes validation."""
    df = load_data()
    assert validate_data(df) is True


def test_validate_data_empty():
    """Test that an empty DataFrame raises a ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="empty"):
        validate_data(empty_df)


def test_validate_data_missing_target_column():
    """Test that a DataFrame missing the target column raises a ValueError."""
    df = load_data()
    df_missing_target = df.drop(columns=[TARGET_COLUMN])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_data(df_missing_target)


def test_validate_data_missing_feature_column():
    """Test that a DataFrame missing a required feature column raises a ValueError."""
    df = load_data()
    df_missing_feature = df.drop(columns=["Age"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_data(df_missing_feature)
