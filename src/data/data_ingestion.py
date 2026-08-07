"""
data_ingestion.py
-----------------
Data ingestion layer for the Bank Customer Churn Prediction project.

Responsibilities
~~~~~~~~~~~~~~~~
1. Load the raw CSV dataset from the path defined in project configuration.
2. Validate that the source file exists before attempting to read it.
3. Rename the original target column (``Exited`` → ``Churn``) so that every
   downstream module works with a consistent column name.
4. Return a clean ``pandas.DataFrame`` ready for validation and preprocessing.

This module intentionally does **not** perform any preprocessing, feature
engineering, encoding, scaling, train/test splitting, or model training.
Those responsibilities belong to dedicated downstream modules.

Usage
~~~~~
    from src.data.data_ingestion import load_data

    df = load_data()
"""

import pandas as pd
from pathlib import Path

from src.config.config import (
    RAW_DATA_PATH,
    ORIGINAL_TARGET_COLUMN,
    TARGET_COLUMN,
)


def load_data(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw churn dataset and prepare it for downstream consumption.

    Parameters
    ----------
    file_path : Path, optional
        Absolute path to the CSV file.
        Defaults to ``RAW_DATA_PATH`` defined in ``src.config.config``.

    Returns
    -------
    pd.DataFrame
        DataFrame with the original target column renamed from
        ``Exited`` to ``Churn``.

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not point to an existing file.
    ValueError
        If the loaded DataFrame is empty (zero rows) or if the expected
        original target column is missing from the dataset.

    Examples
    --------
    >>> from src.data.data_ingestion import load_data
    >>> df = load_data()
    >>> "Churn" in df.columns
    True
    """
    file_path = Path(file_path)

    # --- 1. Verify that the source file exists ---
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'. "
            f"Please ensure the CSV file is placed in the expected location."
        )

    # --- 2. Read the CSV into a DataFrame ---
    df: pd.DataFrame = pd.read_csv(file_path)

    # --- 3. Guard against an empty dataset ---
    if df.empty:
        raise ValueError(
            f"The dataset loaded from '{file_path.name}' contains no rows. "
            f"Please verify the source file."
        )

    # --- 4. Rename the target column for pipeline consistency ---
    if ORIGINAL_TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Expected column '{ORIGINAL_TARGET_COLUMN}' is missing from the "
            f"dataset. Available columns: {list(df.columns)}"
        )

    df = df.rename(columns={ORIGINAL_TARGET_COLUMN: TARGET_COLUMN})

    return df


# =============================================================================
# LOCAL TESTING
# =============================================================================
if __name__ == "__main__":
    try:
        data = load_data()
        print("Dataset loaded successfully.")
        print(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")
        print(data.head())
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
