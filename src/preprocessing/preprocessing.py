"""Data preprocessing module for the Bank Customer Churn Prediction project.

Responsibilities:
- Impute missing values.
- Handle numerical outliers.
- Scale numerical features.
- One-hot encode categorical features.

Important:
The preprocessing transformer is fitted only on the training data
to prevent data leakage.

The train/test split is performed exclusively in main.py.
"""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------------------------
# Create preprocessing transformer
# ---------------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Numerical pipeline
    # -----------------------------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]
    )

    # -----------------------------------------------------------------------
    # Categorical pipeline
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Combine pipelines
    # -----------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Preprocess train and test data
# ---------------------------------------------------------------------------

def preprocess_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    ColumnTransformer,
]:
    """Preprocess training and test datasets.

    The preprocessing transformer is fitted only on X_train.
    X_test is transformed using the fitted transformer.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.

    X_test : pd.DataFrame
        Test features.

    Returns
    -------
    tuple
        X_train_processed :
            Preprocessed training features.

        X_test_processed :
            Preprocessed test features.

        preprocessor :
            Fitted preprocessing transformer.
    """

    # -----------------------------------------------------------------------
    # 1. Validate inputs
    # -----------------------------------------------------------------------

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")

    if X_train.empty:
        raise ValueError("X_train is empty.")

    if X_test.empty:
        raise ValueError("X_test is empty.")

    # -----------------------------------------------------------------------
    # 2. Identify feature types using training data only
    # -----------------------------------------------------------------------

    numerical_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # -----------------------------------------------------------------------
    # 3. Create preprocessing transformer
    # -----------------------------------------------------------------------

    preprocessor = create_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    # -----------------------------------------------------------------------
    # 4. Fit ONLY on training data
    # -----------------------------------------------------------------------

    X_train_processed = preprocessor.fit_transform(X_train)

    # -----------------------------------------------------------------------
    # 5. Transform test data using the fitted transformer
    # -----------------------------------------------------------------------

    X_test_processed = preprocessor.transform(X_test)

    # -----------------------------------------------------------------------
    # 6. Recover transformed feature names
    # -----------------------------------------------------------------------

    feature_names = preprocessor.get_feature_names_out()

    # -----------------------------------------------------------------------
    # 7. Convert transformed arrays back to DataFrames
    # -----------------------------------------------------------------------

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
        preprocessor,
    )