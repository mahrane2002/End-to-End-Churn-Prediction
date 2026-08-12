"""Main entry point for the Bank Customer Churn Prediction project."""

from sklearn.model_selection import train_test_split

from src.config.config import (
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data.data_ingestion import load_data
from src.data.data_validation import validate_data
from src.models.evaluate import evaluate_model
from src.models.predict import predict, predict_proba
from src.models.train import train_model
from src.models.tuning import tune_model
from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.feature_selection import select_features
from src.preprocessing.preprocessing import preprocess_data


def main() -> dict:
    """Run the complete churn prediction pipeline.

    Returns
    -------
    dict
        Dictionary containing the trained model, preprocessing
        transformer, feature selector, predictions, evaluation
        metrics, and tuning results.
    """

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
            f"Target column '{TARGET_COLUMN}' "
            "was not found in the dataset."
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # ==============================================================
    # 4. Train / Test split
    #
    # IMPORTANT:
    # The test set is isolated before feature engineering,
    # preprocessing, feature selection, tuning, and training.
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
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}")

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
        f"X_test after feature engineering: "
        f"{X_test.shape}"
    )

    # ==============================================================
    # 6. Preprocessing
    #
    # The preprocessor is fitted on X_train only.
    # X_test is only transformed.
    # ==============================================================

    print("\n" + "=" * 70)
    print("6. Final preprocessing")
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
        f"X_test after preprocessing: "
        f"{X_test_processed.shape}"
    )

    # ==============================================================
    # 7. Feature Selection
    #
    # IMPORTANT:
    # The selector is fitted ONLY on X_train.
    #
    # X_test is never used to determine which features to keep.
    # ==============================================================

    print("\n" + "=" * 70)
    print("7. Feature selection")
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
        f"X_test after feature selection: "
        f"{X_test_selected.shape}"
    )

    print(
        f"Number of selected features: "
        f"{X_train_selected.shape[1]}"
    )

    # ==============================================================
    # 8. Hyperparameter Tuning
    #
    # IMPORTANT:
    # Tuning is performed AFTER feature selection.
    #
    # Only X_train_selected is used by Optuna.
    # X_test_selected remains untouched.
    # ==============================================================

    print("\n" + "=" * 70)
    print("8. Hyperparameter tuning")
    print("=" * 70)

    best_params, best_score, study = tune_model(
        X_train=X_train_selected,
        y_train=y_train,
        n_trials=50,
    )

    print("\nBest parameters:")

    for parameter, value in best_params.items():
        print(f"  {parameter}: {value}")

    print(
        f"\nBest mean CV ROC-AUC: "
        f"{best_score:.4f}"
    )

    # ==============================================================
    # 9. Final Model Training
    #
    # The model is trained using:
    #   - selected features
    #   - best hyperparameters
    #   - training data only
    #
    # SMOTETomek is applied inside train_model() only to
    # the training data.
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
    # 10. Prediction
    #
    # IMPORTANT:
    # The test set is used for prediction only after all
    # training-related steps are finished.
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
    print(f"Number of predictions: {len(y_pred)}")

    # ==============================================================
    # 11. Model Evaluation
    #
    # evaluate.py receives the predictions generated by predict.py.
    # It does NOT generate predictions itself.
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
    # 12. Return pipeline artifacts
    # ==============================================================

    print("\n" + "=" * 70)
    print("Pipeline completed successfully")
    print("=" * 70)

    return {
        # Final model
        "model": model,

        # Preprocessing and feature selection
        "preprocessor": preprocessor,
        "selector": selector,

        # Hyperparameter tuning
        "best_params": best_params,
        "best_score": best_score,
        "study": study,

        # Test data
        "X_test": X_test_selected,
        "y_test": y_test,

        # Predictions
        "y_pred": y_pred,
        "y_proba": y_proba,

        # Evaluation
        "metrics": metrics,
    }


if __name__ == "__main__":
    main()