import pandas as pd

from src.data.data_validation import validate_data


def test_validate_valid_data():
    df = pd.DataFrame({
        "RowNumber": [1, 2],
        "CustomerId": [1001, 1002],
        "Surname": ["Smith", "Jones"],
        "CreditScore": [650, 700],
        "Geography": ["France", "Spain"],
        "Gender": ["Male", "Female"],
        "Age": [30, 40],
        "Tenure": [5, 3],
        "Balance": [1000, 2000],
        "NumOfProducts": [1, 2],
        "HasCrCard": [1, 0],
        "IsActiveMember": [1, 1],
        "EstimatedSalary": [50000, 60000],
        "Churn": [0, 1],
    })

    assert validate_data(df) is True