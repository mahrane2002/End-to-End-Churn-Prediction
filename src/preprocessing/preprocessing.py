
"""
preprocessing.py
----------------
Preprocessing pipeline for the Bank Customer Churn Prediction project.

Responsibilities:
    - Remove identifier and non-predictive columns.
    - Separate features and target.
    - Split data into training and test sets.
    - Impute missing values.
    - Handle numerical outliers.
    - Scale numerical features.
    - One-hot encode categorical features.
    - Save the fitted preprocessor for later inference.

Important:
    The preprocessor is fitted only on the training data to prevent
    data leakage.

    SMOTETomek is intentionally NOT applied here. It should be applied
    only to the training data inside the model training pipeline.
"""

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engine.outliers import Winsorizer

from src.config.config import (
    ARTIFACT_DIR,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data.data_ingestion import load_data


# ---------------------------------------------------------------------------
# Columns that should not be used as predictive features
# ---------------------------------------------------------------------------

DROP_COLUMNS = [
    "RowNumber",
    "CustomerId",
    "Surname",
]


def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate input features from the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Validated churn dataset.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Features X and target y.
    """


    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Remove identifiers and non-predictive text columns.
    X = X.drop(columns=DROP_COLUMNS, errors="ignore")

    return X, y


def create_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Create the preprocessing pipeline.

    Numerical features:
        1. Median imputation
        2. Winsorization for extreme values
        3. Standard scaling

    Categorical features:
        1. Most-frequent imputation
        2. One-hot encoding

    Parameters
    ----------
    numerical_features : list[str]
        Numerical feature names.

    categorical_features : list[str]
        Categorical feature names.

    Returns
    -------
    ColumnTransformer
        Complete preprocessing transformer.
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "winsorizer",
                Winsorizer(
                    capping_method="iqr",
                    tail="both",
                    fold=1.5,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def preprocess_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Prepare the dataset for model training.

    The train/test split is performed BEFORE fitting the preprocessing
    transformer. This prevents information from the test set from
    influencing imputation, outlier capping, scaling, or encoding.

    Parameters
    ----------
    df : pd.DataFrame
        Validated input dataset.

    Returns
    -------
    Tuple
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        fitted_preprocessor
    """

    # -----------------------------------------------------------------------
    # 1. Separate features and target
    # -----------------------------------------------------------------------

    X, y = split_features_target(df)

    # -----------------------------------------------------------------------
    # 2. Train/test split
    # -----------------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # -----------------------------------------------------------------------
    # 3. Identify feature types
    # -----------------------------------------------------------------------

    numerical_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # -----------------------------------------------------------------------
    # 4. Create preprocessing pipeline
    # -----------------------------------------------------------------------

    preprocessor = create_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    # -----------------------------------------------------------------------
    # 5. Fit ONLY on training data
    # -----------------------------------------------------------------------

    X_train_processed = preprocessor.fit_transform(X_train)

    # -----------------------------------------------------------------------
    # 6. Transform test data using the fitted transformer
    # -----------------------------------------------------------------------

    X_test_processed = preprocessor.transform(X_test)

    # -----------------------------------------------------------------------
    # 7. Recover feature names
    # -----------------------------------------------------------------------

    feature_names = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index,
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index,
    )

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
    )


def save_preprocessor(
    preprocessor: ColumnTransformer,
    output_path: Path = PREPROCESSOR_PATH,
) -> None:
    """
    Save the fitted preprocessing transformer.

    Saving the preprocessor ensures that the exact same transformations
    learned during training can be applied during inference.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, output_path)

    print(f"Preprocessor saved to: {output_path}")


def run_preprocessing():
    """
    Execute the complete preprocessing workflow.
    """

    # Load data through the ingestion module.
    df = load_data()

    # Preprocess data.
    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    ) = preprocess_data(df)

    # Save fitted preprocessor for inference.
    save_preprocessor(preprocessor)

    print("Preprocessing completed successfully.")
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_preprocessing()
```
