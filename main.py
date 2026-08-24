"""Main entry point for the Bank Customer Churn Prediction project."""

from datetime import datetime
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
from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.feature_selection import select_features
from src.preprocessing.preprocessing import preprocess_data
from src.utils.artifact_manager import (
    save_model,
    save_preprocessor,
    save_selector,
    save_metadata,
)


def main(customer_index: int | None = None) -> dict:
    """Run the complete churn prediction pipeline."""

    # ==============================================================
    # 1. Load data
    # ==============================================================

    print("=" * 70)
    print("1. Loading data")
    print("=" * 70)

    df = load_data()

    print(f"Dataset shape: {df.shape}")

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
            f"Target column '{TARGET_COLUMN}' was not found in the dataset."
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

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

    # ==============================================================
    # 5. Feature Engineering
    # ==============================================================

    print("\n" + "=" * 70)
    print("5. Feature engineering")
    print("=" * 70)

    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)

    print(f"X_train after feature engineering: {X_train.shape}")
    print(f"X_test after feature engineering:  {X_test.shape}")

    # ==============================================================
    # 6. Hyperparameter Tuning
    #
    # IMPORTANT:
    # Do NOT perform feature selection here beforehand.
    #
    # tune_model() performs:
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
    # Feature selection is therefore fitted independently
    # inside every CV fold.
    # ==============================================================

    print("\n" + "=" * 70)
    print("6. Hyperparameter tuning")
    print("=" * 70)

    best_params, best_score, study = tune_model(
        X_train=X_train,
        y_train=y_train,
        n_trials=50,
    )

    print("\nBest parameters:")
    for parameter, value in best_params.items():
        print(f"  {parameter}: {value}")

    print(f"\nBest mean CV ROC-AUC: {best_score:.4f}")

    # ==============================================================
    # 7. Final preprocessing
    #
    # This preprocessing is fitted on the complete X_train.
    # X_test is only transformed.
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

    print(f"X_train after preprocessing: {X_train_processed.shape}")
    print(f"X_test after preprocessing:  {X_test_processed.shape}")

    # ==============================================================
    # 8. Final Feature Selection
    #
    # IMPORTANT:
    # This is the FINAL selector.
    #
    # It is fitted on the complete X_train only.
    # X_test is only transformed.
    #
    # This selector is NOT used to perform CV.
    # CV already performed its own selection inside each fold.
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

    print(f"X_train after feature selection: {X_train_selected.shape}")
    print(f"X_test after feature selection:  {X_test_selected.shape}")
    print(f"Number of selected features:     {X_train_selected.shape[1]}")

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

    print("Final XGBoost model trained successfully.")

    # ==============================================================
    # 9.5 Save Artifacts
    # ==============================================================

    print("\n" + "=" * 70)
    print("9.5 Saving artifacts")
    print("=" * 70)

    save_model(model)
    save_preprocessor(preprocessor)
    save_selector(selector)

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
    "training_date": datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    ),
}
    save_metadata(metadata)
    print("All artifacts saved successfully.")

    # ==============================================================
    # 10. Prediction
    # ==============================================================

    print("\n" + "=" * 70)
    print("10. Prediction")
    print("=" * 70)

    y_pred = predict(
        model=model,
        X=X_test_selected,
    )

    y_proba = predict_proba(
        model=model,
        X=X_test_selected,
    )

    print("Predictions generated successfully.")

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
    # 12. SHAP Explainability
    #
    # IMPORTANT:
    # SHAP is applied to the final trained XGBoost model.
    #
    # The explanation data is X_test_selected, which corresponds
    # exactly to the features received by the final XGBoost model.
    #
    # SMOTETomek is NOT applied to X_test.
    #
    # The prediction threshold remains 0.5.
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

    print("Global SHAP explanations generated successfully.")

    customer_explanation = None

    if customer_index is not None:
        customer_explanation = explain_customer(
            model=model,
            explainer=explainer,
            X_test=X_test_selected,
            client_index=customer_index,
            threshold=0.5,
        )

        print(f"SHAP explanation generated for customer index {customer_index}.")

    # ==============================================================
    # 13. Return artifacts
    # ==============================================================

    print("\n" + "=" * 70)
    print("Pipeline completed successfully")
    print("=" * 70)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "selector": selector,
        "best_params": best_params,
        "best_score": best_score,
        "study": study,
        "X_test": X_test_selected,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": metrics,
        "shap_explainer": explainer,
        "shap_results": shap_results,
        "customer_explanation": customer_explanation,
    }


if __name__ == "__main__":
    main()