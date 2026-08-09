
"""Data preprocessing module for the Bank Customer Churn Prediction project.

Responsibilities:
- Remove identifier and non-predictive columns.
- Separate features and target.
- Split data into training and test sets.
- Impute missing values.
- Handle numerical outliers.
- Scale numerical features.
- One-hot encode categorical features.

Important:
The preprocessing transformer is fitted only on the training data
to prevent data leakage.

"""

from typing import Tuple

import pandas as pd

from feature_engine.outliers import Winsorizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.config import (
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


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
    """Separate features from the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Validated churn dataset.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Features X and target y.
    """

    # Separate features and target.
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Remove identifiers and non-predictive columns.
    X = X.drop(columns=DROP_COLUMNS, errors="ignore")

    return X, y


def create_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Create the preprocessing transformer.

    Numerical features:
    - Median imputation
    - Winsorization
    - Standard scaling

    Categorical features:
    - Most-frequent imputation
    - One-hot encoding

    Parameters
    ----------
    numerical_features : list[str]
        Names of numerical features.

    categorical_features : list[str]
        Names of categorical features.

    Returns
    -------
    ColumnTransformer
        Configured preprocessing transformer.
    """

    # Pipeline for numerical features.
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

    # Pipeline for categorical features.
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

    # Apply the appropriate pipeline to each feature type.
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
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    ColumnTransformer,
]:
    """Preprocess the dataset for model training.

    The train/test split is performed before fitting the preprocessing
    transformer. This prevents information from the test set from
    influencing imputation, outlier handling, scaling, or encoding.

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
        fitted_preprocessor.
    """

    # -----------------------------------------------------------------------
    # 1. Separate features and target
    # -----------------------------------------------------------------------

    X, y = split_features_target(df)

    # -----------------------------------------------------------------------
    # 2. Split data into training and test sets
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
    # 4. Create the preprocessing transformer
    # -----------------------------------------------------------------------

    preprocessor = create_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    # -----------------------------------------------------------------------
    # 5. Fit only on training data
    # -----------------------------------------------------------------------

    X_train_processed = preprocessor.fit_transform(X_train)

    # -----------------------------------------------------------------------
    # 6. Transform test data using the fitted transformer
    # -----------------------------------------------------------------------

    X_test_processed = preprocessor.transform(X_test)

    # -----------------------------------------------------------------------
    # 7. Recover transformed feature names
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
