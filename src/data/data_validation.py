"""
data_validation.py
------------------
Data validation layer for the Bank Customer Churn Prediction project.

Responsibilities
~~~~~~~~~~~~~~~~
1. Perform quality checks on a DataFrame produced by data_ingestion.py.
2. Validate DataFrame size, schema, data types, missing values, target column,
   and duplicate rows.
3. Generate a structured validation report and export it as JSON.

This module intentionally does NOT perform data ingestion, preprocessing, feature
engineering, scaling, encoding, or model training.

Usage
~~~~~
    from src.data.data_ingestion import load_data
    from src.data.data_validation import DataValidator

    df = load_data()
    validator = DataValidator()
    report = validator.validate(df)
    validator.save_validation_report(report)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd

from src.config.config import ARTIFACT_DIR, TARGET_COLUMN


# Default list of expected columns in raw dataset after data ingestion (Exited -> Churn)
DEFAULT_EXPECTED_COLUMNS: List[str] = [
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


class DataValidator:
    """
    Data quality validator for dataset DataFrames.

    Performs multi-step quality validation including size checks, schema checks,
    data type analysis, missing value calculations, target integrity checks,
    and duplicate detection.

    Parameters
    ----------
    expected_columns : Optional[List[str]], optional
        List of column names expected in the DataFrame.
        Defaults to DEFAULT_EXPECTED_COLUMNS.
    target_column : str, optional
        Name of the target column. Defaults to TARGET_COLUMN from config.
    max_missing_pct : float, optional
        Maximum allowed percentage of missing values per column (0.0 to 1.0).
        Defaults to 0.05 (5%).
    report_path : Optional[Union[str, Path]], optional
        Destination path for saving JSON validation reports.
        Defaults to artifacts/validation/validation_report.json.

    Raises
    ------
    ValueError
        If parameters are misconfigured (e.g. invalid threshold).
    """

    def __init__(
        self,
        expected_columns: Optional[List[str]] = None,
        target_column: str = TARGET_COLUMN,
        max_missing_pct: float = 0.05,
        report_path: Optional[Union[str, Path]] = None,
    ) -> None:
        if not (0.0 <= max_missing_pct <= 1.0):
            raise ValueError("max_missing_pct must be a float between 0.0 and 1.0.")

        self.expected_columns: List[str] = (
            list(expected_columns)
            if expected_columns is not None
            else list(DEFAULT_EXPECTED_COLUMNS)
        )
        self.target_column: str = target_column
        self.max_missing_pct: float = max_missing_pct

        if report_path is None:
            self.report_path: Path = (
                ARTIFACT_DIR / "validation" / "validation_report.json"
            )
        else:
            self.report_path = Path(report_path)

    def validate_dataframe_size(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate dataset dimensions and verify it is non-empty.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to validate.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing row count, column count, empty flag, and pass status.
        """
        num_rows: int = len(df)
        num_cols: int = len(df.columns)
        is_empty: bool = df.empty or num_rows == 0

        return {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "is_empty": is_empty,
            "passed": not is_empty,
        }

    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate DataFrame schema against expected columns.

        Detects missing and unexpected extra columns.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to validate.

        Returns
        -------
        Dict[str, Any]
            Dictionary with lists of missing and extra columns and pass status.
        """
        actual_cols: Set[str] = set(df.columns)
        expected_cols: Set[str] = set(self.expected_columns)

        missing_columns: List[str] = sorted(list(expected_cols - actual_cols))
        extra_columns: List[str] = sorted(list(actual_cols - expected_cols))
        passed: bool = len(missing_columns) == 0

        return {
            "expected_columns": self.expected_columns,
            "missing_columns": missing_columns,
            "extra_columns": extra_columns,
            "passed": passed,
        }

    def validate_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Inspect and categorize column data types.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to inspect.

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping columns to dtypes, numeric columns, categorical columns.
        """
        detected_types: Dict[str, str] = {
            col: str(dtype) for col, dtype in df.dtypes.items()
        }

        numeric_columns: List[str] = df.select_dtypes(
            include=["number"]
        ).columns.tolist()
        categorical_columns: List[str] = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        return {
            "detected_types": detected_types,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "passed": True,
        }

    def validate_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate missing value counts and percentages, comparing against a threshold.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to check.

        Returns
        -------
        Dict[str, Any]
            Summary of missing values count, percentages, and threshold compliance.
        """
        total_rows: int = len(df)
        missing_counts: pd.Series = df.isnull().sum()
        missing_values_per_column: Dict[str, int] = missing_counts.to_dict()

        missing_percentage_per_column: Dict[str, float] = {}
        columns_above_threshold: List[str] = []

        if total_rows > 0:
            for col, count in missing_counts.items():
                pct: float = float(count / total_rows)
                missing_percentage_per_column[col] = round(pct, 4)
                if pct > self.max_missing_pct:
                    columns_above_threshold.append(col)

        total_missing_values: int = int(missing_counts.sum())
        passed: bool = len(columns_above_threshold) == 0

        return {
            "total_missing_values": total_missing_values,
            "missing_values_per_column": missing_values_per_column,
            "missing_percentage_per_column": missing_percentage_per_column,
            "columns_above_threshold": columns_above_threshold,
            "max_missing_pct_threshold": self.max_missing_pct,
            "passed": passed,
        }

    def validate_target_column(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Verify target column presence and check for valid binary values [0, 1].

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to check.

        Returns
        -------
        Dict[str, Any]
            Target validation status, unique values, and invalid values if any.
        """
        if self.target_column not in df.columns:
            return {
                "target_column": self.target_column,
                "exists": False,
                "unique_values": [],
                "allowed_values": [0, 1],
                "invalid_values": [],
                "passed": False,
            }

        unique_vals: List[Any] = df[self.target_column].dropna().unique().tolist()
        allowed_set: Set[Any] = {0, 1, 0.0, 1.0}
        invalid_values: List[Any] = [
            val for val in unique_vals if val not in allowed_set
        ]

        passed: bool = len(invalid_values) == 0

        return {
            "target_column": self.target_column,
            "exists": True,
            "unique_values": [
                int(v) if isinstance(v, (int, float)) and float(v).is_integer() else v
                for v in unique_vals
            ],
            "allowed_values": [0, 1],
            "invalid_values": invalid_values,
            "passed": passed,
        }

    def validate_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Count duplicate rows in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to check.

        Returns
        -------
        Dict[str, Any]
            Count and percentage of duplicate rows.
        """
        num_rows: int = len(df)
        duplicate_count: int = int(df.duplicated().sum())
        duplicate_pct: float = (
            round(float(duplicate_count / num_rows), 4) if num_rows > 0 else 0.0
        )

        return {
            "duplicate_count": duplicate_count,
            "duplicate_percentage": duplicate_pct,
            "passed": True,
        }

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute full multi-step data validation on the input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame provided by data_ingestion.py.

        Returns
        -------
        Dict[str, Any]
            Comprehensive validation report dictionary.

        Raises
        ------
        TypeError
            If df is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected a pandas DataFrame, but received {type(df).__name__}."
            )

        errors: List[str] = []

        # 1. Size Validation
        dataframe_size_res = self.validate_dataframe_size(df)
        if not dataframe_size_res["passed"]:
            errors.append("DataFrame is empty (0 rows).")

        # 2. Schema Validation
        schema_res = self.validate_schema(df)
        if not schema_res["passed"]:
            errors.append(
                f"Missing expected columns: {schema_res['missing_columns']}"
            )

        # 3. Data Types Validation
        data_types_res = self.validate_data_types(df)

        # 4. Missing Values Validation
        missing_val_res = self.validate_missing_values(df)
        if not missing_val_res["passed"]:
            errors.append(
                f"Columns exceeding max missing threshold ({self.max_missing_pct * 100}%): "
                f"{missing_val_res['columns_above_threshold']}"
            )

        # 5. Target Column Validation
        target_res = self.validate_target_column(df)
        if not target_res["exists"]:
            errors.append(
                f"Target column '{self.target_column}' is missing from DataFrame."
            )
        elif not target_res["passed"]:
            errors.append(
                f"Target column '{self.target_column}' contains invalid values: "
                f"{target_res['invalid_values']}. Allowed values are [0, 1]."
            )

        # 6. Duplicates Validation
        duplicates_res = self.validate_duplicates(df)

        status: str = "passed" if len(errors) == 0 else "failed"

        report: Dict[str, Any] = {
            "status": status,
            "dataframe_size": dataframe_size_res,
            "schema": schema_res,
            "data_types": data_types_res,
            "missing_values": missing_val_res,
            "target_column": target_res,
            "duplicates": duplicates_res,
            "errors": errors,
        }

        return report

    def save_validation_report(
        self, results: Dict[str, Any], output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Save the validation report dictionary as a formatted JSON file using pathlib.

        Parameters
        ----------
        results : Dict[str, Any]
            Validation report produced by validate().
        output_path : Optional[Union[str, Path]], optional
            Target JSON file path. Defaults to self.report_path.

        Returns
        -------
        Path
            Path to the written JSON file.
        """
        destination: Path = (
            Path(output_path) if output_path is not None else self.report_path
        )
        destination.parent.mkdir(parents=True, exist_ok=True)

        with open(destination, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        return destination


if __name__ == "__main__":
    from src.data.data_ingestion import load_data

    raw_df = load_data()
    validator = DataValidator()
    validation_results = validator.validate(raw_df)
    saved_file = validator.save_validation_report(validation_results)
    print(json.dumps(validation_results, indent=4))
