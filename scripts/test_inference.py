import pandas as pd

from src.utils.artifact_manager import load_inference_pipeline


def main():
    print("=" * 70)
    print("TESTING PRODUCTION INFERENCE PIPELINE")
    print("=" * 70)

    pipeline = load_inference_pipeline()

    customer = pd.DataFrame([
        {
            "CreditScore": 650,
            "Geography": "France",
            "Gender": "Male",
            "Age": 40,
            "Tenure": 5,
            "Balance": 100000,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 60000,
        }
    ])

    print("\nRAW CUSTOMER:")
    print(customer)

    prediction = pipeline.predict(customer)
    probability = pipeline.predict_proba(customer)

    print("\nRESULT:")
    print(f"Prediction: {int(prediction[0])}")
    print(f"Churn probability: {float(probability[0]):.4f}")


if __name__ == "__main__":
    main()