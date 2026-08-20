import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_raw_data():
    """Provides a small DataFrame mimicking the raw Churn_Modelling.csv file."""
    data = {
        "RowNumber": list(range(1, 16)),
        "CustomerId": [15634600 + i for i in range(1, 16)],
        "Surname": [
            "Hargrave", "Hill", "Onio", "Boni", "Mitchell", "Chu", "Bartlett",
            "Obinna", "He", "H", "Bearce", "Andrews", "Kay", "Chin", "Scott"
        ],
        "CreditScore": [619, 608, 502, 699, 850, 645, 822, 376, 501, 684, 528, 497, 476, 549, 700],
        "Geography": ["France", "Spain", "France", "France", "Spain", "Spain", "France", "Germany", "France", "France", "France", "Spain", "France", "France", "Germany"],
        "Gender": ["Female", "Female", "Female", "Female", "Female", "Male", "Male", "Female", "Male", "Male", "Male", "Male", "Female", "Female", "Male"],
        "Age": [42, 41, 42, 39, 43, 44, 50, 29, 44, 27, 31, 24, 34, 25, 40],
        "Tenure": [2, 1, 8, 1, 2, 8, 7, 4, 4, 2, 6, 3, 10, 5, 5],
        "Balance": [0.0, 83807.86, 159660.8, 0.0, 125510.82, 113755.78, 0.0, 115046.74, 142051.07, 134603.88, 102016.72, 0.0, 0.0, 0.0, 120000.0],
        "NumOfProducts": [1, 1, 3, 2, 1, 2, 2, 4, 2, 1, 2, 2, 2, 2, 1],
        "HasCrCard": [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        "IsActiveMember": [1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
        "EstimatedSalary": [101348.88, 112542.58, 113931.57, 93826.63, 79084.1, 149756.71, 10062.8, 119346.88, 74940.5, 71725.73, 80181.12, 76390.01, 26260.98, 190857.79, 90000.0],
        "Exited": [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_df(sample_raw_data):
    """Provides a valid ingested DataFrame where Exited has been renamed to Churn."""
    df = sample_raw_data.copy()
    df = df.rename(columns={"Exited": "Churn"})
    return df

@pytest.fixture
def large_sample_df():
    """Generates a larger, unique dataset with 100 rows that passes validation."""
    np.random.seed(42)
    n = 100
    data = {
        "RowNumber": list(range(1, n + 1)),
        "CustomerId": [15634600 + i for i in range(1, n + 1)],
        "Surname": [f"Surname_{i}" for i in range(n)],
        "CreditScore": np.random.randint(400, 850, size=n),
        "Geography": np.random.choice(["France", "Spain", "Germany"], size=n),
        "Gender": np.random.choice(["Female", "Male"], size=n),
        "Age": np.random.randint(18, 80, size=n),
        "Tenure": np.random.randint(0, 11, size=n),
        "Balance": np.random.uniform(0.0, 200000.0, size=n),
        "NumOfProducts": np.random.randint(1, 4, size=n),
        "HasCrCard": np.random.randint(0, 2, size=n),
        "IsActiveMember": np.random.randint(0, 2, size=n),
        "EstimatedSalary": np.random.uniform(10000.0, 200000.0, size=n),
        "Churn": np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    }
    return pd.DataFrame(data)
