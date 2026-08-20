import pytest
import pandas as pd
from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.preprocessing import preprocess_data
from src.preprocessing.feature_selection import select_features

def test_engineer_features(sample_df):
    # Arrange: Use sample data (excluding target)
    X = sample_df.drop(columns=["Churn"])

    # Act: Engineer new features
    X_engineered = engineer_features(X)

    # Assert: Verify new columns are created successfully
    assert isinstance(X_engineered, pd.DataFrame)
    expected_new_cols = ["BalancePerAge", "ProductsPerTenure", "IsActiveAndHasCard", "AgeGroup", "HasBalance", "MultipleProducts"]
    for col in expected_new_cols:
        assert col in X_engineered.columns

    # Verify input validation
    with pytest.raises(TypeError):
        engineer_features([1, 2, 3]) # type: ignore
    with pytest.raises(ValueError):
        engineer_features(pd.DataFrame())

def test_preprocess_data(sample_df):
    # Arrange: Add engineered features first to mimic real flow
    X = engineer_features(sample_df.drop(columns=["Churn"]))
    X_train = X.iloc[:10]
    X_test = X.iloc[10:]

    # Act: Run preprocessing
    X_train_proc, X_test_proc, preprocessor = preprocess_data(X_train, X_test)

    # Assert: Check structure, output types, and feature shapes
    assert isinstance(X_train_proc, pd.DataFrame)
    assert isinstance(X_test_proc, pd.DataFrame)
    assert preprocessor is not None
    assert X_train_proc.shape[0] == 10
    assert X_test_proc.shape[0] == 5
    # One-hot encoding should create binary columns for Geography and Gender
    assert any(col.startswith("categorical__Geography_") for col in X_train_proc.columns)

def test_select_features(sample_df):
    # Arrange: Preprocess some data
    X = engineer_features(sample_df.drop(columns=["Churn"]))
    y = sample_df["Churn"]
    X_train = X.iloc[:10]
    y_train = y.iloc[:10]
    X_test = X.iloc[10:]
    
    X_train_proc, X_test_proc, _ = preprocess_data(X_train, X_test)

    # Act: Select the top k features (k=5)
    X_train_sel, X_test_sel, selector = select_features(
        X_train=X_train_proc,
        y_train=y_train,
        X_test=X_test_proc,
        k=5
    )

    # Assert: Check number of features selected and identifier column removal
    assert isinstance(X_train_sel, pd.DataFrame)
    assert X_train_sel.shape[1] <= 5
    # Identifier columns (like numerical__RowNumber or CustomerId) should not be present
    assert "numerical__RowNumber" not in X_train_sel.columns
    assert "RowNumber" not in X_train_sel.columns
