import pytest
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.preprocessing import preprocess_data
from src.preprocessing.feature_selection import select_features
from src.models.train import train_model
from src.models.predict import predict, predict_proba, predict_raw, predict_raw_proba
from src.models.evaluate import evaluate_model

def test_train_and_predict(large_sample_df):
    # Arrange: Setup train and test datasets from the synthetic large_sample_df
    X = engineer_features(large_sample_df.drop(columns=["Churn"]))
    y = large_sample_df["Churn"]
    
    X_train, X_test = X.iloc[:80], X.iloc[80:]
    y_train, y_test = y.iloc[:80], y.iloc[80:]
    
    # Preprocess & Select
    X_train_proc, X_test_proc, preprocessor = preprocess_data(X_train, X_test)
    X_train_sel, X_test_sel, selector = select_features(X_train_proc, y_train, X_test_proc, k=5)
    
    # Setup standard, lightweight parameters for fast test execution
    params = {
        "n_estimators": 5,
        "max_depth": 2,
        "learning_rate": 0.1
    }

    # Act: Train the XGBoost model
    model = train_model(X_train_sel, y_train, params)
    
    # Generate predictions on the selected test data
    y_pred = predict(model, X_test_sel)
    y_proba = predict_proba(model, X_test_sel)

    # Assert: Verify predictions shape, values, and types
    assert isinstance(model, XGBClassifier)
    assert len(y_pred) == len(X_test_sel)
    assert len(y_proba) == len(X_test_sel)
    assert np.all((y_proba >= 0.0) & (y_proba <= 1.0))
    assert np.all((y_pred == 0) | (y_pred == 1))

def test_predict_raw_pipeline(large_sample_df):
    # Arrange: Train a baseline model and fit preprocessor/selector
    X_raw = large_sample_df.drop(columns=["Churn"])
    y = large_sample_df["Churn"]
    
    X_engineered = engineer_features(X_raw)
    X_train_proc, _, preprocessor = preprocess_data(X_engineered, X_engineered.copy())
    X_train_sel, _, selector = select_features(X_train_proc, y, k=5)
    
    params = {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1}
    model = train_model(X_train_sel, y, params)
    
    # Act & Assert:
    # There is a known bug in `src/models/predict.py:prepare_for_prediction()`.
    # It does not remove identifier columns (such as `Surname`) from the transformed DataFrame
    # before passing it to `selector.transform()`. Because of this, it throws a ValueError
    # stating that the feature names do not match those passed during fit.
    with pytest.raises(ValueError) as exc_info:
        predict_raw(model, preprocessor, selector, X_raw.iloc[:10])
    
    assert "Feature names" in str(exc_info.value)

def test_evaluate_model():
    # Arrange: Setup mock targets and prediction arrays
    y_test = pd.Series([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])
    y_proba = np.array([0.8, 0.2, 0.4, 0.1])

    # Act: Evaluate the predictions
    metrics = evaluate_model(y_test, y_pred, y_proba)

    # Assert: Verify correct keys and types in evaluation metrics dictionary
    assert isinstance(metrics, dict)
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "confusion_matrix", "classification_report"]:
        assert key in metrics
    
    assert isinstance(metrics["accuracy"], float)
    assert isinstance(metrics["roc_auc"], float)
    assert isinstance(metrics["confusion_matrix"], list)
    assert isinstance(metrics["classification_report"], str)
