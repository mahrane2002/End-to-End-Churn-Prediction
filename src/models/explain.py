"""
SHAP-based explainability module for the churn prediction project.

This module provides:
- SHAP TreeExplainer creation
- Global SHAP explanations
- Customer-level SHAP explanations for API inference
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from src.config.config import ARTIFACT_DIR, RANDOM_STATE


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

SHAP_DIR = ARTIFACT_DIR / "shap"

DEFAULT_BACKGROUND_SIZE = 200
DEFAULT_THRESHOLD = 0.5


# -------------------------------------------------------------------
# SHAP Explainer
# -------------------------------------------------------------------

def create_tree_explainer(
    model: XGBClassifier,
    background_data: pd.DataFrame,
    background_size: int = DEFAULT_BACKGROUND_SIZE,
) -> shap.TreeExplainer:
    """
    Create a SHAP TreeExplainer for the final XGBoost model.

    Parameters
    ----------
    model : XGBClassifier
        Final trained XGBoost model.

    background_data : pd.DataFrame
        Training data after preprocessing and feature selection.

        IMPORTANT:
        SMOTETomek must NOT be applied to this data.

    background_size : int, default=200
        Maximum number of observations used as SHAP background.

    Returns
    -------
    shap.TreeExplainer
        Configured SHAP TreeExplainer.
    """

    if not isinstance(model, XGBClassifier):
        raise TypeError(
            "model must be an XGBClassifier."
        )

    if not isinstance(background_data, pd.DataFrame):
        raise TypeError(
            "background_data must be a pandas DataFrame."
        )

    if background_data.empty:
        raise ValueError(
            "background_data must not be empty."
        )

    if background_size <= 0:
        raise ValueError(
            "background_size must be greater than zero."
        )

    background_size = min(
        background_size,
        len(background_data),
    )

    background = background_data.sample(
        n=background_size,
        random_state=RANDOM_STATE,
    )

    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
        model_output="probability",
    )

    return explainer


# -------------------------------------------------------------------
# Global explanation
# -------------------------------------------------------------------

def explain_global(
    explainer: shap.TreeExplainer,
    X_test: pd.DataFrame,
    output_dir: Path = SHAP_DIR,
    max_display: int = 15,
) -> pd.DataFrame:
    """
    Generate global SHAP explanations for the test set.

    Generates:
    - SHAP feature importance CSV
    - SHAP feature importance bar plot

    Parameters
    ----------
    explainer : shap.TreeExplainer
        SHAP explainer created for the final model.

    X_test : pd.DataFrame
        Test data after preprocessing and feature selection.

    output_dir : Path
        Directory where SHAP artifacts are saved.

    max_display : int
        Maximum number of features displayed.

    Returns
    -------
    pd.DataFrame
        Feature importance table sorted by mean absolute SHAP value.
    """

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(
            "X_test must be a pandas DataFrame."
        )

    if X_test.empty:
        raise ValueError(
            "X_test must not be empty."
        )

    if max_display <= 0:
        raise ValueError(
            "max_display must be greater than zero."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Calculate SHAP values
    # ---------------------------------------------------------------

    explanation = explainer(X_test)

    shap_values = np.asarray(
        explanation.values
    )

    # ---------------------------------------------------------------
    # Handle possible SHAP output shapes
    # ---------------------------------------------------------------

    if shap_values.ndim == 3:
        # Binary/multiclass compatibility.
        # For churn probability, use the positive class.
        shap_values = shap_values[:, :, 1]

    if shap_values.ndim != 2:
        raise ValueError(
            "Unexpected SHAP values shape: "
            f"{shap_values.shape}"
        )

    # ---------------------------------------------------------------
    # Mean absolute SHAP importance
    # ---------------------------------------------------------------

    mean_abs_shap = np.abs(
        shap_values
    ).mean(axis=0)

    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values(
        by="mean_abs_shap",
        ascending=False,
    )

    # ---------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------

    importance_path = (
        output_dir
        / "shap_feature_importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Save bar plot
    # ---------------------------------------------------------------

    plt.figure()

    shap.plots.bar(
        explanation,
        max_display=max_display,
        show=False,
    )

    plt.tight_layout()

    importance_plot_path = (
        output_dir
        / "shap_feature_importance.png"
    )

    plt.savefig(
        importance_plot_path,
        bbox_inches="tight",
        dpi=300,
    )

    plt.close()

    return importance


# -------------------------------------------------------------------
# Customer-level explanation
# -------------------------------------------------------------------

def explain_prediction(
    model: XGBClassifier,
    explainer: shap.TreeExplainer,
    X_ready: pd.DataFrame,
    original_data: pd.DataFrame,
    top_k: int = 5,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict[str, Any]]:
    """
    Explain the prediction of one customer using SHAP.

    Parameters
    ----------
    model : XGBClassifier
        Final trained XGBoost model.

    explainer : shap.TreeExplainer
        SHAP explainer for the final model.

    X_ready : pd.DataFrame
        Customer after:
            Feature Engineering
            -> Preprocessing
            -> Feature Selection

        This is the exact feature space used by XGBoost.

    original_data : pd.DataFrame
        Original raw customer data.

        This is used ONLY to return human-readable
        feature values in the API response.

    top_k : int, default=5
        Number of most important features to return.

    threshold : float, default=0.5
        Classification threshold.

    Returns
    -------
    list[dict]
        Top SHAP feature explanations.

    Example
    -------
    [
        {
            "feature": "Age",
            "value": 45,
            "shap_value": 0.31,
            "effect": "increases_churn_probability"
        }
    ]
    """

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    if not isinstance(model, XGBClassifier):
        raise TypeError(
            "model must be an XGBClassifier."
        )

    if not isinstance(X_ready, pd.DataFrame):
        raise TypeError(
            "X_ready must be a pandas DataFrame."
        )

    if not isinstance(original_data, pd.DataFrame):
        raise TypeError(
            "original_data must be a pandas DataFrame."
        )

    if X_ready.empty:
        raise ValueError(
            "X_ready must not be empty."
        )

    if original_data.empty:
        raise ValueError(
            "original_data must not be empty."
        )

    if len(X_ready) != 1:
        raise ValueError(
            "X_ready must contain exactly one customer."
        )

    if len(original_data) != 1:
        raise ValueError(
            "original_data must contain exactly one customer."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0.0 and 1.0."
        )

    # ---------------------------------------------------------------
    # Verify feature compatibility
    # ---------------------------------------------------------------

    if X_ready.shape[1] == 0:
        raise ValueError(
            "X_ready contains no features."
        )

    # ---------------------------------------------------------------
    # Model probability
    # ---------------------------------------------------------------

    churn_probability = float(
        model.predict_proba(
            X_ready
        )[0, 1]
    )

    # ---------------------------------------------------------------
    # Prediction using explicit threshold
    # ---------------------------------------------------------------

    prediction = int(
        churn_probability >= threshold
    )

    # ---------------------------------------------------------------
    # SHAP explanation
    # ---------------------------------------------------------------

    shap_explanation = explainer(
        X_ready
    )

    shap_values = np.asarray(
        shap_explanation.values
    )

    # ---------------------------------------------------------------
    # Handle SHAP output shape
    # ---------------------------------------------------------------

    if shap_values.ndim == 3:
        # Binary/multiclass compatibility.
        shap_values = shap_values[0, :, 1]

    elif shap_values.ndim == 2:
        shap_values = shap_values[0]

    elif shap_values.ndim == 1:
        shap_values = shap_values

    else:
        raise ValueError(
            "Unexpected SHAP values shape: "
            f"{shap_values.shape}"
        )

    shap_values = np.asarray(
        shap_values
    ).reshape(-1)

    # ---------------------------------------------------------------
    # Verify SHAP feature count
    # ---------------------------------------------------------------

    if len(shap_values) != X_ready.shape[1]:
        raise ValueError(
            "Number of SHAP values does not match "
            "the number of model features. "
            f"SHAP={len(shap_values)}, "
            f"features={X_ready.shape[1]}"
        )

    # ---------------------------------------------------------------
    # Build transformed feature contribution table
    # ---------------------------------------------------------------

    contributions = pd.DataFrame(
        {
            "feature": X_ready.columns,
            "shap_value": shap_values,
            "absolute_shap_value": np.abs(
                shap_values
            ),
        }
    )

    # ---------------------------------------------------------------
    # Sort by absolute SHAP importance
    # ---------------------------------------------------------------

    contributions = contributions.sort_values(
        by="absolute_shap_value",
        ascending=False,
    )

    # ---------------------------------------------------------------
    # Select top K
    # ---------------------------------------------------------------

    top_features = contributions.head(
        min(top_k, len(contributions))
    )

    # ---------------------------------------------------------------
    # Map transformed features to original values
    # ---------------------------------------------------------------
    #
    # IMPORTANT:
    #
    # X_ready contains transformed features.
    #
    # original_data contains raw customer values.
    #
    # We therefore try to recover the original value
    # whenever the transformed feature name corresponds
    # to an original feature.
    #
    # If a feature was engineered/transformed and cannot
    # be mapped directly, we return None instead of
    # pretending that the transformed value is the raw value.
    # ---------------------------------------------------------------

    original_columns = set(
        original_data.columns
    )

    results = []

    for _, row in top_features.iterrows():

        feature_name = str(
            row["feature"]
        )

        shap_value = float(
            row["shap_value"]
        )

        # -----------------------------------------------------------
        # Try to identify original feature
        # -----------------------------------------------------------

        original_feature = None

        # Direct match
        if feature_name in original_columns:
            original_feature = feature_name

        else:
            # Common sklearn ColumnTransformer names:
            #
            # num__Age
            # cat__Geography_France
            #
            # Try removing transformer prefix.
            if "__" in feature_name:
                candidate = feature_name.split(
                    "__",
                    1
                )[1]

                if candidate in original_columns:
                    original_feature = candidate

            # -------------------------------------------------------
            # Handle one-hot encoded categorical features
            # -------------------------------------------------------

            if original_feature is None:

                for column in original_data.columns:

                    if feature_name.startswith(
                        f"{column}_"
                    ):
                        original_feature = column
                        break

        # -----------------------------------------------------------
        # Retrieve original value
        # -----------------------------------------------------------

        if original_feature is not None:

            value = original_data.iloc[0][
                original_feature
            ]

            # Convert NumPy values to native Python values
            if isinstance(
                value,
                np.generic,
            ):
                value = value.item()

        else:

            # Do NOT expose transformed/scaled values
            # as if they were raw customer values.
            value = None

        # -----------------------------------------------------------
        # Direction of impact
        # -----------------------------------------------------------

        if shap_value >= 0:
            effect = (
                "increases_churn_probability"
            )
        else:
            effect = (
                "decreases_churn_probability"
            )

        results.append(
            {
                "feature": feature_name,
                "value": value,
                "shap_value": shap_value,
                "effect": effect,
            }
        )

    return results


# -------------------------------------------------------------------
# Offline customer explanation
# -------------------------------------------------------------------

def explain_customer(
    model: XGBClassifier,
    explainer: shap.TreeExplainer,
    X_test: pd.DataFrame,
    client_index: Any,
    threshold: float = DEFAULT_THRESHOLD,
    output_dir: Path = SHAP_DIR,
    max_display: int = 15,
) -> pd.DataFrame:
    """
    Generate a detailed offline SHAP explanation
    for a specific customer.

    This function is intended for analysis and artifact generation.

    The customer must already be represented in the
    same feature space used by XGBoost.
    """

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(
            "X_test must be a pandas DataFrame."
        )

    if X_test.empty:
        raise ValueError(
            "X_test must not be empty."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0.0 and 1.0."
        )

    if client_index not in X_test.index:
        raise KeyError(
            f"Client index '{client_index}' "
            "was not found in X_test."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Select exactly one customer
    # ---------------------------------------------------------------

    X_client = X_test.loc[
        [client_index]
    ]

    # ---------------------------------------------------------------
    # Model probability
    # ---------------------------------------------------------------

    churn_probability = float(
        model.predict_proba(
            X_client
        )[0, 1]
    )

    prediction = int(
        churn_probability >= threshold
    )

    # ---------------------------------------------------------------
    # SHAP explanation
    # ---------------------------------------------------------------

    explanation = explainer(
        X_client
    )

    client_explanation = explanation[0]

    shap_values = np.asarray(
        client_explanation.values
    ).reshape(-1)

    feature_values = (
        X_client.iloc[0]
        .to_numpy()
    )

    contributions = pd.DataFrame(
        {
            "feature": X_test.columns,
            "feature_value": feature_values,
            "shap_value": shap_values,
            "absolute_shap_value": np.abs(
                shap_values
            ),
        }
    ).sort_values(
        by="absolute_shap_value",
        ascending=False,
    )

    contributions["effect"] = np.where(
        contributions["shap_value"] >= 0,
        "increases_churn_probability",
        "decreases_churn_probability",
    )

    contributions["client_index"] = (
        client_index
    )

    contributions["churn_probability"] = (
        churn_probability
    )

    contributions["prediction"] = (
        prediction
    )

    # ---------------------------------------------------------------
    # Save contributions
    # ---------------------------------------------------------------

    csv_path = (
        output_dir
        / f"client_{client_index}_shap.csv"
    )

    contributions.to_csv(
        csv_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Waterfall plot
    # ---------------------------------------------------------------

    plt.figure()

    shap.plots.waterfall(
        client_explanation,
        max_display=max_display,
        show=False,
    )

    plt.tight_layout()

    plot_path = (
        output_dir
        / f"client_{client_index}_waterfall.png"
    )

    plt.savefig(
        plot_path,
        bbox_inches="tight",
        dpi=300,
    )

    plt.close()

    print("\n" + "=" * 70)
    print("CUSTOMER SHAP EXPLANATION")
    print("=" * 70)

    print(
        f"Client index       : {client_index}"
    )

    print(
        f"Churn probability  : "
        f"{churn_probability:.4f}"
    )

    print(
        f"Threshold          : "
        f"{threshold:.2f}"
    )

    print(
        f"Final prediction   : "
        f"{prediction}"
    )

    print("\nTop feature contributions:")

    print(
        contributions[
            [
                "feature",
                "feature_value",
                "shap_value",
                "effect",
            ]
        ]
        .head(max_display)
        .to_string(index=False)
    )

    print("=" * 70)

    return contributions