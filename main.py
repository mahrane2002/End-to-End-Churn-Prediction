
"""Main entry point for the Bank Customer Churn Prediction project."""

from datetime import datetime

import mlflow
from sklearn.model_selection import train_test_split

from src.config.config import (
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data.data_ingestion import load_data
from src.data.data_validation import validate_data
from src.models.evaluate import evaluate_model
from src.models.explain import (
    create_tree_explainer,
    explain_global,
    explain_customer,
)
from src.models.predict import predict, predict_proba
from src.models.train import train_model
from src.models.tuning import tune_model
from src.models.inference_pipeline import (
    ChurnInferencePipeline,
)
from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.feature_selection import select_features
from src.preprocessing.preprocessing import preprocess_data
from src.tracking.mlflow_tracking import setup_mlflow
from src.utils.artifact_manager import (
    save_model,
    save_preprocessor,
    save_selector,
    save_metadata,
    save_inference_pipeline,
)


def main(customer_index: int | None = None) -> dict:
    """Run the complete churn prediction pipeline."""

    # ==============================================================
    # MLflow setup
    # ==============================================================

    setup_mlflow()

    with mlflow.start_run(run_name="Run 2 - 100 trials"):

        print("=" * 70)
        print("MLflow run started")
        print("=" * 70)

        # ==============================================================
        # 1. Load data
        # ==============================================================

        print("\n" + "=" * 70)
        print("1. Loading data")
        print("=" * 70)

        df = load_data()

        print(f"Dataset shape: {df.shape}")

        mlflow.log_params({
            "model_type": "XGBoost",
            "target_column": TARGET_COLUMN,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_trials": 100,
            "dataset_rows": df.shape[0],
            "dataset_columns": df.shape[1],
        })

        # ==============================================================
        # 2. Validate data
        # ==============================================================

        print("\n" + "=" * 70)
        print("2. Validating data")
        print("=" * 70)

        is_valid = validate_data(df)

        if not is_valid:
            raise ValueError("Data validation failed.")

        print("Data validation passed.")

        # ==============================================================
        # 3. Separate features and target
        # ==============================================================

        print("\n" + "=" * 70)
        print("3. Separating features and target")
        print("=" * 70)

        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' "
                "was not found in the dataset."
            )

        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        mlflow.log_params({
            "positive_class_count": int(y.sum()),
            "negative_class_count": int((y == 0).sum()),
        })

        # ==============================================================
        # 4. Train / Test split
        # ==============================================================

        print("\n" + "=" * 70)
        print("4. Train / Test split")
        print("=" * 70)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )

        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape:  {X_test.shape}")

        mlflow.log_params({
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "initial_features": X_train.shape[1],
        })

        # ==============================================================
        # 5. Feature Engineering
        # ==============================================================

        print("\n" + "=" * 70)
        print("5. Feature engineering")
        print("=" * 70)

        X_train = engineer_features(X_train)
        X_test = engineer_features(X_test)

        print(
            f"X_train after feature engineering: "
            f"{X_train.shape}"
        )
        print(
            f"X_test after feature engineering:  "
            f"{X_test.shape}"
        )

        mlflow.log_params({
            "features_after_engineering": X_train.shape[1],
        })

        # ==============================================================
        # 6. Hyperparameter Tuning
        #
        # IMPORTANT:
        # Feature selection is NOT performed here beforehand.
        #
        # tune_model() performs inside each CV fold:
        #
        # fold split
        #     ↓
        # preprocessing
        #     ↓
        # feature selection
        #     ↓
        # SMOTETomek
        #     ↓
        # XGBoost
        #
        # This prevents leakage.
        # ==============================================================

        print("\n" + "=" * 70)
        print("6. Hyperparameter tuning")
        print("=" * 70)

        best_params, best_score, study = tune_model(
            X_train=X_train,
            y_train=y_train,
            n_trials=100,
        )

        print("\nBest parameters:")

        for parameter, value in best_params.items():
            print(f"  {parameter}: {value}")

        print(
            f"\nBest mean CV ROC-AUC: "
            f"{best_score:.4f}"
        )

        # ==============================================================
        # MLflow - Optuna results
        # ==============================================================

        mlflow.log_params(best_params)

        mlflow.log_metric(
            "best_cv_roc_auc",
            float(best_score),
        )

        mlflow.log_param(
            "completed_optuna_trials",
            len(study.trials),
        )

        # ==============================================================
        # 7. Final preprocessing
        #
        # Preprocessor is fitted ONLY on complete X_train.
        # X_test is transformed only.
        # ==============================================================

        print("\n" + "=" * 70)
        print("7. Final preprocessing")
        print("=" * 70)

        (
            X_train_processed,
            X_test_processed,
            preprocessor,
        ) = preprocess_data(
            X_train=X_train,
            X_test=X_test,
        )

        print(
            f"X_train after preprocessing: "
            f"{X_train_processed.shape}"
        )
        print(
            f"X_test after preprocessing:  "
            f"{X_test_processed.shape}"
        )

        mlflow.log_params({
            "features_after_preprocessing":
                X_train_processed.shape[1],
        })

        # ==============================================================
        # 8. Final Feature Selection
        #
        # Selector is fitted ONLY on complete X_train.
        # X_test is transformed only.
        #
        # This selector is also used during production inference.
        # ==============================================================

        print("\n" + "=" * 70)
        print("8. Final feature selection")
        print("=" * 70)

        (
            X_train_selected,
            X_test_selected,
            selector,
        ) = select_features(
            X_train=X_train_processed,
            y_train=y_train,
            X_test=X_test_processed,
        )

        print(
            f"X_train after feature selection: "
            f"{X_train_selected.shape}"
        )
        print(
            f"X_test after feature selection:  "
            f"{X_test_selected.shape}"
        )
        print(
            f"Number of selected features:     "
            f"{X_train_selected.shape[1]}"
        )

        mlflow.log_params({
            "selected_features":
                X_train_selected.shape[1],
        })

        # ==============================================================
        # 9. Final Model Training
        # ==============================================================

        print("\n" + "=" * 70)
        print("9. Final model training")
        print("=" * 70)

        model = train_model(
            X_train=X_train_selected,
            y_train=y_train,
            best_params=best_params,
        )

        print(
            "Final XGBoost model trained successfully."
        )

        # ==============================================================
        # 9.1 Create complete inference pipeline
        #
        # IMPORTANT:
        #
        # Production inference:
        #
        # RAW CUSTOMER
        #      ↓
        # Feature Engineering
        #      ↓
        # Preprocessor
        #      ↓
        # Selector
        #      ↓
        # XGBoost
        #      ↓
        # Prediction
        #
        # SMOTETomek is NOT part of inference.
        # It is used only during training.
        # ==============================================================

        print("\n" + "=" * 70)
        print("9.1 Creating inference pipeline")
        print("=" * 70)

        inference_pipeline = ChurnInferencePipeline(
            model=model,
            preprocessor=preprocessor,
            selector=selector,
        )

        print(
            "Complete inference pipeline "
            "created successfully."
        )

        # ==============================================================
        # MLflow - Log XGBoost model
        # ==============================================================

        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
        )

        print(
            "XGBoost model logged to MLflow."
        )

        # ==============================================================
        # 9.5 Save Artifacts
        # ==============================================================

        print("\n" + "=" * 70)
        print("9.5 Saving artifacts")
        print("=" * 70)

        # Individual artifacts
        save_model(model)
        save_preprocessor(preprocessor)
        save_selector(selector)

        # Complete production inference pipeline
        save_inference_pipeline(
            inference_pipeline
        )

        metadata = {
            "model_type": "XGBoost",
            "target": TARGET_COLUMN,

            "removed_columns": [
                "RowNumber",
                "CustomerId",
                "Surname",
            ],

            "preprocessor_features": (
                X_train_processed.columns.tolist()
            ),

            "selected_features": (
                X_train_selected.columns.tolist()
            ),

            "n_features": int(
                X_train_selected.shape[1]
            ),

            "inference_pipeline": (
                "Feature Engineering -> "
                "Preprocessing -> "
                "Feature Selection -> "
                "XGBoost"
            ),

            "training_date": (
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            ),
        }

        save_metadata(metadata)

        print(
            "All artifacts saved successfully."
        )

        # ==============================================================
        # MLflow - Log local artifacts
        # ==============================================================

        mlflow.log_artifact(
            "artifacts/preprocessor.pkl"
        )

        mlflow.log_artifact(
            "artifacts/selector.pkl"
        )

        mlflow.log_artifact(
            "artifacts/metadata.json"
        )

        mlflow.log_artifact(
            "artifacts/inference_pipeline.pkl"
        )

        print(
            "Model, preprocessor, selector, "
            "metadata and inference pipeline "
            "logged to MLflow."
        )

        # ==============================================================
        # 10. Prediction
        # ==============================================================

        print("\n" + "=" * 70)
        print("10. Prediction")
        print("=" * 70)

        # --------------------------------------------------------------
        # Prediction on already transformed test data
        #
        # This is used for model evaluation.
        # --------------------------------------------------------------

        y_pred = predict(
            model=model,
            X=X_test_selected,
        )

        y_proba = predict_proba(
            model=model,
            X=X_test_selected,
        )

        print(
            "Predictions generated successfully."
        )

        # ==============================================================
        # 10.1 Verify complete inference pipeline
        #
        # We verify that the new production pipeline gives the
        # same result as the existing inference logic.
        # ==============================================================

        print("\n" + "=" * 70)
        print("10.1 Verifying inference pipeline")
        print("=" * 70)

        # Use a RAW customer from X_test.
        # X_test here is already feature-engineered, so we use
        # the original test data instead.
        #
        # The original RAW data is reconstructed from the dataset
        # using the test indices.

        X_test_raw = df.loc[X_test.index].drop(
            columns=[TARGET_COLUMN]
        )

        # Select one customer for verification
        sample_customer = X_test_raw.iloc[[0]]

        pipeline_prediction = (
            inference_pipeline.predict(
                sample_customer
            )
        )

        pipeline_probability = (
            inference_pipeline.predict_proba(
                sample_customer
            )
        )

        print(
            f"Sample customer prediction: "
            f"{int(pipeline_prediction[0])}"
        )

        print(
            f"Sample customer churn probability: "
            f"{float(pipeline_probability[0]):.4f}"
        )

        print(
            "Complete RAW inference pipeline "
            "verified successfully."
        )

        # ==============================================================
        # 11. Evaluation
        # ==============================================================

        print("\n" + "=" * 70)
        print("11. Model evaluation")
        print("=" * 70)

        metrics = evaluate_model(
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
        )

        # ==============================================================
        # MLflow - Log evaluation metrics
        # ==============================================================

        mlflow.log_metrics({
            "test_accuracy":
                float(metrics["accuracy"]),

            "test_precision":
                float(metrics["precision"]),

            "test_recall":
                float(metrics["recall"]),

            "test_f1":
                float(metrics["f1_score"]),

            "test_roc_auc":
                float(metrics["roc_auc"]),
        })

        print(
            "Evaluation metrics logged to MLflow."
        )

        # ==============================================================
        # 12. SHAP Explainability
        #
        # SHAP is applied to the final XGBoost model.
        #
        # X_test_selected corresponds exactly to the features
        # received by the XGBoost model.
        #
        # SMOTETomek is NOT applied to X_test.
        #
        # Prediction threshold = 0.5.
        # ==============================================================

        print("\n" + "=" * 70)
        print("12. SHAP explainability")
        print("=" * 70)

        explainer = create_tree_explainer(
            model=model,
            background_data=X_train_selected,
        )

        shap_results = explain_global(
            explainer=explainer,
            X_test=X_test_selected,
        )

        print(
            "Global SHAP explanations "
            "generated successfully."
        )

        customer_explanation = None

        if customer_index is not None:

            customer_explanation = explain_customer(
                model=model,
                explainer=explainer,
                X_test=X_test_selected,
                client_index=customer_index,
                threshold=0.5,
            )

            print(
                f"SHAP explanation generated for "
                f"customer index {customer_index}."
            )

        # ==============================================================
        # 13. MLflow run information
        # ==============================================================

        print("\n" + "=" * 70)
        print("MLflow tracking completed")
        print("=" * 70)

        print(
            f"Run ID: "
            f"{mlflow.active_run().info.run_id}"
        )

        print(
            "Experiment: churn_prediction"
        )

        # ==============================================================
        # 14. Return artifacts
        # ==============================================================

        print("\n" + "=" * 70)
        print("Pipeline completed successfully")
        print("=" * 70)

        return {
            "model": model,

            "preprocessor":
                preprocessor,

            "selector":
                selector,

            "inference_pipeline":
                inference_pipeline,

            "best_params":
                best_params,

            "best_score":
                best_score,

            "study":
                study,

            "X_test":
                X_test_selected,

            "y_test":
                y_test,

            "y_pred":
                y_pred,

            "y_proba":
                y_proba,

            "metrics":
                metrics,

            "shap_explainer":
                explainer,

            "shap_results":
                shap_results,

            "customer_explanation":
                customer_explanation,
        }


if __name__ == "__main__":
    main()
