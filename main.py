"""Main entry point for the Bank Customer Churn Prediction project."""

from src.config.config import (
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data.data_ingestion import load_data
from src.data.data_validation import validate_data
from src.models.train import train_model
from src.models.tuning import tune_model
from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.preprocessing import preprocess_data

from sklearn.model_selection import train_test_split


def main() -> dict:
    """Run the complete churn prediction pipeline.

    Returns
    -------
    dict
        Dictionary containing the trained model, preprocessing
        transformer, test data, and tuning results.
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
    # preprocessing, tuning, and training.
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
    # 5. Feature engineering
    #
    # Feature engineering is applied separately to train and test.
    #
    # No statistics are learned from the test set.
    # ==============================================================

    print("\n" + "=" * 70)
    print("5. Feature engineering")
    print("=" * 70)

    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)

    print(f"X_train after feature engineering: {X_train.shape}")
    print(f"X_test after feature engineering:  {X_test.shape}")

    # ==============================================================
    # 6. Hyperparameter tuning
    #
    # The tuning module performs:
    #
    #   Stratified K-Fold
    #       ↓
    #   preprocessing fitted on train fold
    #       ↓
    #   preprocessing transform validation fold
    #       ↓
    #   SMOTETomek on train fold only
    #       ↓
    #   XGBoost
    #       ↓
    #   ROC-AUC
    #
    # No feature selection is performed.
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
    # The preprocessor is now fitted on the COMPLETE training set.
    #
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

    print(
        f"X_train after preprocessing: "
        f"{X_train_processed.shape}"
    )

    print(
        f"X_test after preprocessing:  "
        f"{X_test_processed.shape}"
    )

    # ==============================================================
    # 8. Final model training
    #
    # train_model applies SMOTETomek ONLY to X_train.
    #
    # The test set is NOT used here.
    # ==============================================================

    print("\n" + "=" * 70)
    print("8. Final model training")
    print("=" * 70)

    model = train_model(
        X_train=X_train_processed,
        y_train=y_train,
        best_params=best_params,
    )

    print("Final XGBoost model trained successfully.")

    # ==============================================================
    # 9. Return pipeline artifacts
    # ==============================================================

    print("\n" + "=" * 70)
    print("Pipeline completed")
    print("=" * 70)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "best_params": best_params,
        "best_score": best_score,
        "study": study,
        "X_test": X_test_processed,
        "y_test": y_test,
    }


if __name__ == "__main__":
    main()