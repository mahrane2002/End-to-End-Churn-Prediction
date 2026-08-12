"""Main pipeline for the Bank Customer Churn Prediction project."""

from sklearn.model_selection import train_test_split

from src.config.config import RANDOM_STATE, TEST_SIZE

from src.data.data_ingestion import load_data
from src.data.data_validation import validate_data

from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.preprocessing import preprocess_data
from src.preprocessing.feature_selection import select_features

from src.models.tuning import tune_model
from src.models.train import train_model


def main() -> None:
    """Run the complete churn prediction training pipeline."""

    # ==============================================================
    # 1. DATA INGESTION
    # ==============================================================

    df = load_data()

    # ==============================================================
    # 2. DATA VALIDATION
    # ==============================================================

    is_valid = validate_data(df)

    if not is_valid:
        raise ValueError(
            "Data validation failed."
        )

    # ==============================================================
    # 3. SEPARATE FEATURES AND TARGET
    # ==============================================================

    X = df.drop(
        columns=["Churn"]
    )

    y = df["Churn"]

    # ==============================================================
    # 4. TRAIN / TEST SPLIT
    # ==============================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # ==============================================================
    # 5. FEATURE ENGINEERING
    # ==============================================================

    X_train = engineer_features(
        X_train
    )

    X_test = engineer_features(
        X_test
    )

    # ==============================================================
    # 6. PREPROCESSING
    # ==============================================================

    (
        X_train_processed,
        X_test_processed,
        preprocessor,
    ) = preprocess_data(
        X_train,
        X_test,
    )

    # ==============================================================
    # 7. FEATURE SELECTION
    # ==============================================================

    (
        X_train_selected,
        X_test_selected,
        selector,
    ) = select_features(
        X_train=X_train_processed,
        y_train=y_train,
        X_test=X_test_processed,
    )

    # ==============================================================
    # 8. HYPERPARAMETER TUNING
    # ==============================================================

    best_params, best_score, study = tune_model(
        X_train=X_train_selected,
        y_train=y_train,
        n_trials=50,
    )

    print("Best CV ROC-AUC:", best_score)
    print("Best parameters:", best_params)

    # ==============================================================
    # 9. FINAL TRAINING
    # ==============================================================

    model = train_model(
        X_train=X_train_selected,
        y_train=y_train,
        best_params=best_params,
    )

    print("Final model trained successfully.")

    # ==============================================================
    # 10. RETURN OBJECTS
    # ==============================================================

    return {
        "model": model,
        "preprocessor": preprocessor,
        "selector": selector,
        "best_params": best_params,
        "best_score": best_score,
        "study": study,
        "X_test": X_test_selected,
        "y_test": y_test,
    }


if __name__ == "__main__":
    main()