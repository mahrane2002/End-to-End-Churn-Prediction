import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from main import main

def test_pipeline_integration(large_sample_df):
    # Arrange:
    # 1. Prepare raw data (containing "Exited" instead of "Churn")
    raw_df = large_sample_df.copy()
    raw_df = raw_df.rename(columns={"Churn": "Exited"})

    # Define mock returns
    mock_best_params = {
        "n_estimators": 5,
        "max_depth": 2,
        "learning_rate": 0.1
    }

    # Act & Assert: Patch all file writers, data ingestion, Optuna tuning, validation, and explainers
    # to run instantly without creating files in the production paths.
    with patch("src.data.data_ingestion.load_data", return_value=raw_df), \
         patch("src.models.tuning.tune_model", return_value=(mock_best_params, 0.85, None)), \
         patch("main.validate_data", return_value=True), \
         patch("main.save_model") as mock_save_model, \
         patch("main.save_preprocessor") as mock_save_preprocessor, \
         patch("main.save_selector") as mock_save_selector, \
         patch("main.save_metadata") as mock_save_metadata, \
         patch("main.create_tree_explainer", return_value=MagicMock()) as mock_create_explainer, \
         patch("main.explain_global", return_value=pd.DataFrame()) as mock_explain_global:

        # Run the full pipeline
        results = main(customer_index=None)

        # Assert: Verify all pipeline components returned valid outputs
        assert isinstance(results, dict)
        assert results["model"] is not None
        assert results["preprocessor"] is not None
        assert results["selector"] is not None
        assert results["best_params"] == mock_best_params
        assert results["best_score"] == 0.85
        assert len(results["y_pred"]) == len(results["X_test"])
        assert len(results["y_proba"]) == len(results["X_test"])
        assert "accuracy" in results["metrics"]
        assert isinstance(results["shap_results"], pd.DataFrame)

        # Assert: Verify saving functions were called correctly
        mock_save_model.assert_called_once_with(results["model"])
        mock_save_preprocessor.assert_called_once_with(results["preprocessor"])
        mock_save_selector.assert_called_once_with(results["selector"])
        mock_save_metadata.assert_called_once()
        mock_create_explainer.assert_called_once()
        mock_explain_global.assert_called_once()
