"""SHAP-based explainability module for the churn prediction project."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from src.config.config import ARTIFACT_DIR, RANDOM_STATE


SHAP_DIR = ARTIFACT_DIR / "shap"
DEFAULT_BACKGROUND_SIZE = 200
DEFAULT_THRESHOLD = 0.5


def create_tree_explainer(
    model: XGBClassifier,
    background_data: pd.DataFrame,
    background_size: int = DEFAULT_BACKGROUND_SIZE,
) -> shap.TreeExplainer:
    """Create a SHAP TreeExplainer for the final XGBoost model.

    The explainer is configured to explain churn probabilities rather
    than raw XGBoost margins.

    Parameters
    ----------
    model : XGBClassifier
        Final trained XGBoost model.

    background_data : pd.DataFrame
        Training data after preprocessing and feature selection.
        SMOTETomek must NOT be applied to this data.

    background_size : int, default=200
        Maximum number of background observations.

    Returns
    -------
    shap.TreeExplainer
        Configured SHAP TreeExplainer.
    """

    if not isinstance(model, XGBClassifier):
        raise TypeError("model must be an XGBClassifier.")

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

    return shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
        model_output="probability",
    )


def explain_global(
    explainer: shap.TreeExplainer,
    X_test: pd.DataFrame,
    output_dir: Path = SHAP_DIR,
    max_display: int = 15,
) -> pd.DataFrame:
    """Generate global SHAP explanations for the test set.

    Generates:
    - SHAP feature importance bar plot.
    - SHAP beeswarm/summary plot.
    - CSV containing mean absolute SHAP importance.

    Parameters
    ----------
    explainer : shap.TreeExplainer
        SHAP explainer created for the final model.

    X_test : pd.DataFrame
        Test data after preprocessing and feature selection.

    output_dir : Path, default=SHAP_DIR
        Directory where SHAP artifacts are saved.

    max_display : int, default=15
        Maximum number of features displayed in plots.

    Returns
    -------
    pd.DataFrame
        Feature importance table sorted by mean absolute SHAP value.
    """

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")

    if X_test.empty:
        raise ValueError("X_test must not be empty.")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    explanation = explainer(X_test)

    shap_values = explanation.values

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values(
        by="mean_abs_shap",
        ascending=False,
    )

    importance_path = (
        output_dir / "shap_feature_importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False,
    )

    # --------------------------------------------------------------
    # SHAP feature importance bar plot
    # --------------------------------------------------------------

    plt.figure()

    shap.plots.bar(
        explanation,
        max_display=max_display,
        show=False,
    )

    plt.tight_layout()

    importance_plot_path = (
        output_dir / "shap_feature_importance.png"
    )

    plt.savefig(
        importance_plot_path,
        bbox_inches="tight",
        dpi=300,
    )

    plt.close()

    return importance


def explain_customer(
    model: XGBClassifier,
    explainer: shap.TreeExplainer,
    X_test: pd.DataFrame,
    client_index: Any,
    threshold: float = DEFAULT_THRESHOLD,
    output_dir: Path = SHAP_DIR,
    max_display: int = 15,
) -> pd.DataFrame:
    """Explain one customer using SHAP.

    The customer must already be represented in the same feature space
    used by the final XGBoost model.

    Parameters
    ----------
    model : XGBClassifier
        Final trained XGBoost model.

    explainer : shap.TreeExplainer
        SHAP TreeExplainer for the final model.

    X_test : pd.DataFrame
        Test data after preprocessing and feature selection.

    client_index : Any
        Index label of the customer to explain.

    threshold : float, default=0.5
        Classification threshold.

    output_dir : Path, default=SHAP_DIR
        Directory where the waterfall plot is saved.

    max_display : int, default=15
        Maximum number of features shown in the waterfall plot.

    Returns
    -------
    pd.DataFrame
        Feature-level SHAP contributions for the selected customer.
    """

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")

    if X_test.empty:
        raise ValueError("X_test must not be empty.")

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

    # --------------------------------------------------------------
    # Select exactly one customer
    # --------------------------------------------------------------

    X_client = X_test.loc[[client_index]]

    # --------------------------------------------------------------
    # Model probability
    # --------------------------------------------------------------

    churn_probability = float(
        model.predict_proba(X_client)[0, 1]
    )

    prediction = int(
        churn_probability >= threshold
    )

    # --------------------------------------------------------------
    # SHAP explanation
    # --------------------------------------------------------------

    explanation = explainer(X_client)

    client_explanation = explanation[0]

    shap_values = np.asarray(
        client_explanation.values
    ).reshape(-1)

    feature_values = X_client.iloc[0].to_numpy()

    contributions = pd.DataFrame(
        {
            "feature": X_test.columns,
            "feature_value": feature_values,
            "shap_value": shap_values,
            "absolute_shap_value": np.abs(shap_values),
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

    contributions["client_index"] = client_index

    contributions["churn_probability"] = (
        churn_probability
    )

    contributions["prediction"] = prediction

    # --------------------------------------------------------------
    # Save contributions
    # --------------------------------------------------------------

    csv_path = (
        output_dir
        / f"client_{client_index}_shap.csv"
    )

    contributions.to_csv(
        csv_path,
        index=False,
    )

    # --------------------------------------------------------------
    # Waterfall plot
    # --------------------------------------------------------------

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

    print(f"Client index       : {client_index}")
    print(
        f"Churn probability  : "
        f"{churn_probability:.4f}"
    )
    print(f"Threshold          : {threshold:.2f}")
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
        ].head(max_display).to_string(
            index=False
        )
    )

    print("=" * 70)

    return contributions