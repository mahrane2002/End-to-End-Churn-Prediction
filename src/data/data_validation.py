"""Data validation module for the churn prediction project."""

import json
from pathlib import Path
import pandas as pd

from src.config.config import ARTIFACT_DIR, TARGET_COLUMN


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
    """Validate the dataset before preprocessing.

    Checks:
    - Dataset is not empty.
    - Expected columns are present.
    - Target column exists.
    - Target contains only 0 and 1.
    - No duplicate rows.
    - Missing values are reported.
    - Age > 0.
    - Tenure >= 0.
    - NumOfProducts > 0.
    - CreditScore within reasonable range [300, 850].

    Writes a report to artifacts/validation/validation_report.json.

    Returns:
        True if all critical validation checks pass.

    Raises:
        ValueError if any validation check fails.
    """

    checks = {}
    is_valid = True
    errors = []

    # 1. Check that the dataset is not empty.
    if df.empty:
        checks["not_empty"] = {"status": "FAILED", "message": "Dataset is empty"}
        is_valid = False
        errors.append("Dataset is empty")
    else:
        checks["not_empty"] = {"status": "PASSED"}

    # 2. Check that all expected columns are present.
    missing_columns = list(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing_columns:
        checks["columns_present"] = {"status": "FAILED", "message": f"missing required columns: {missing_columns}"}
        is_valid = False
        errors.append(f"missing required columns: {missing_columns}")
    else:
        checks["columns_present"] = {"status": "PASSED"}

    # 3. Check that the target column exists.
    if TARGET_COLUMN not in df.columns:
        checks["target_present"] = {"status": "FAILED", "message": f"missing required columns: {TARGET_COLUMN}"}
        is_valid = False
        errors.append(f"missing required columns: {TARGET_COLUMN}")
    else:
        checks["target_present"] = {"status": "PASSED"}

    # 4. Check that the target contains only binary values.
    if TARGET_COLUMN in df.columns:
        target_values = set(df[TARGET_COLUMN].dropna().unique())
        if not target_values.issubset({0, 1}):
            checks["target_binary"] = {"status": "FAILED", "message": f"Target values are not binary: {target_values}"}
            is_valid = False
            errors.append(f"Target values are not binary: {target_values}")
        else:
            checks["target_binary"] = {"status": "PASSED"}

    # 5. Check for duplicate rows.
    if not df.empty:
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count > 0:
            checks["no_duplicates"] = {"status": "FAILED", "message": f"Found {duplicate_count} duplicate rows"}
            is_valid = False
            errors.append(f"Found {duplicate_count} duplicate rows")
        else:
            checks["no_duplicates"] = {"status": "PASSED"}

    # 6. Report missing values.
    if not df.empty:
        missing_values = df.isnull().sum()
        missing_count = int(missing_values.sum())
        checks["missing_values"] = {
            "status": "PASSED" if missing_count == 0 else "WARNING",
            "missing_count_per_column": missing_values[missing_values > 0].to_dict()
        }

    # 7. Check Data Types
    if not df.empty:
        data_types = df.dtypes.astype(str).to_dict()
        checks["data_types"] = {"status": "PASSED", "types": data_types}

    # 8. Check Age > 0
    if "Age" in df.columns:
        invalid_age_count = int((df["Age"] <= 0).sum())
        if invalid_age_count > 0:
            checks["valid_age"] = {"status": "FAILED", "message": f"Found {invalid_age_count} rows with Age <= 0"}
            is_valid = False
            errors.append(f"Found {invalid_age_count} rows with Age <= 0")
        else:
            checks["valid_age"] = {"status": "PASSED"}

    # 9. Check Tenure >= 0
    if "Tenure" in df.columns:
        invalid_tenure_count = int((df["Tenure"] < 0).sum())
        if invalid_tenure_count > 0:
            checks["valid_tenure"] = {"status": "FAILED", "message": f"Found {invalid_tenure_count} rows with Tenure < 0"}
            is_valid = False
            errors.append(f"Found {invalid_tenure_count} rows with Tenure < 0")
        else:
            checks["valid_tenure"] = {"status": "PASSED"}

    # 10. Check NumOfProducts > 0
    if "NumOfProducts" in df.columns:
        invalid_products_count = int((df["NumOfProducts"] <= 0).sum())
        if invalid_products_count > 0:
            checks["valid_products"] = {"status": "FAILED", "message": f"Found {invalid_products_count} rows with NumOfProducts <= 0"}
            is_valid = False
            errors.append(f"Found {invalid_products_count} rows with NumOfProducts <= 0")
        else:
            checks["valid_products"] = {"status": "PASSED"}

    # 11. Check CreditScore within [300, 850]
    if "CreditScore" in df.columns:
        invalid_credit_count = int(((df["CreditScore"] < 300) | (df["CreditScore"] > 850)).sum())
        if invalid_credit_count > 0:
            checks["valid_credit_score"] = {
                "status": "FAILED",
                "message": f"Found {invalid_credit_count} rows with CreditScore outside [300, 850]"
            }
            is_valid = False
            errors.append(f"Found {invalid_credit_count} rows with CreditScore outside [300, 850]")
        else:
            checks["valid_credit_score"] = {"status": "PASSED"}

    # Write report
    report = {
        "is_valid": is_valid,
        "dataset_shape": list(df.shape) if not df.empty else [0, 0],
        "checks": checks,
        "errors": errors
    }

    report_dir = ARTIFACT_DIR / "validation"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "validation_report.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print(f"Validation report saved to {report_path}")

    if not is_valid:
        raise ValueError(f"Validation failed: {'; '.join(errors)}")

    print("Data validation passed.")
    return True
