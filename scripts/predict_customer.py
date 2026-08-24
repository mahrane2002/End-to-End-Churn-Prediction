"""Predict churn for a new raw customer."""

import sys

import pandas as pd

from src.config.config import TARGET_COLUMN
from src.models.predict import (
    predict_raw,
    predict_raw_proba,
)
from src.utils.artifact_manager import (
    load_model,
    load_preprocessor,
    load_selector,
)


def main() -> None:
    print("=" * 70)
    print("CUSTOMER CHURN PREDICTION")
    print("=" * 70)

    # --------------------------------------------------------------
    # 1. Load artifacts
    # --------------------------------------------------------------

    print("\nLoading artifacts...")

    model = load_model()
    preprocessor = load_preprocessor()
    selector = load_selector()

    print("[OK] model.pkl")
    print("[OK] preprocessor.pkl")
    print("[OK] selector.pkl")

    # --------------------------------------------------------------
    # 2. New RAW customer
    # --------------------------------------------------------------

    customer = {
        "RowNumber": 10001,
        "CustomerId": 15634602,
        "Surname": "Smith",
        "CreditScore": 619,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 2,
        "Balance": 0.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 101348.88,
    }

    customer_id = customer.get(
        "CustomerId",
        "N/A",
    )

    customer_df = pd.DataFrame(
        [customer]
    )

    # --------------------------------------------------------------
    # 3. Raw inference
    # --------------------------------------------------------------

    print("\nRunning inference...")

    prediction = predict_raw(
        model=model,
        preprocessor=preprocessor,
        selector=selector,
        X=customer_df,
    )

    probability = predict_raw_proba(
        model=model,
        preprocessor=preprocessor,
        selector=selector,
        X=customer_df,
    )

    # --------------------------------------------------------------
    # 4. Results
    # --------------------------------------------------------------

    predicted_class = int(
        prediction[0]
    )

    churn_probability = float(
        probability[0]
    )

    label = (
        "CHURN"
        if predicted_class == 1
        else "NO CHURN"
    )

    print("\n" + "=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)

    print(
        f"Customer ID       : {customer_id}"
    )

    print(
        f"Churn probability : "
        f"{churn_probability:.4f}"
    )

    print(
        f"Churn probability : "
        f"{churn_probability * 100:.2f}%"
    )

    print(
        f"Prediction        : {label}"
    )

    print(
        f"Target convention : "
        f"{TARGET_COLUMN}=1 means churn"
    )

    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\n[ERROR] Prediction failed:\n{exc}"
        )
        sys.exit(1)