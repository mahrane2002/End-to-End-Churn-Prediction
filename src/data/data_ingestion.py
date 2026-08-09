"""Data ingestion module for loading and initial preparation of churn dataset."""

from pathlib import Path
from typing import Union

import pandas as pd

from src.config.config import ORIGINAL_TARGET_COLUMN, RAW_DATA_PATH, TARGET_COLUMN


def load_data(file_path: Union[str, Path] = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw churn CSV dataset and rename target column to standard name."""
    df = pd.read_csv(file_path)
    if ORIGINAL_TARGET_COLUMN in df.columns:
        df = df.rename(columns={ORIGINAL_TARGET_COLUMN: TARGET_COLUMN})
    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Data loaded successfully with shape: {df.shape}")
