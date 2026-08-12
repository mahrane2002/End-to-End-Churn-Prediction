"""Model evaluation module for the Bank Customer Churn Prediction project."""

from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """
    Evaluate the final churn prediction model on the test set.

    The test set must remain completely untouched during:
        - feature engineering
        - preprocessing fitting
        - feature selection fitting
        - hyperparameter tuning
        - model training

    Parameters
    ----------
    model : Any
        Trained XGBoost classification model.

    X_test : pd.DataFrame
        Test features after the same preprocessing and feature
        selection transformations used for the training data.

    y_test : pd.Series
        True target values for the test set.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the main classification metrics,
        confusion matrix, and classification report.
    """

    # ------------------------------------------------------------------
    # 1. Generate predictions
    # ------------------------------------------------------------------

    y_pred = model.predict(X_test)

    # XGBoost returns probabilities for the positive class
    y_proba = model.predict_proba(X_test)[:, 1]

    # ------------------------------------------------------------------
    # 2. Calculate metrics
    # ------------------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_proba,
    )

    # ------------------------------------------------------------------
    # 3. Confusion matrix
    # ------------------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    # ------------------------------------------------------------------
    # 4. Classification report
    # ------------------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )

    # ------------------------------------------------------------------
    # 5. Store results
    # ------------------------------------------------------------------

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    # ------------------------------------------------------------------
    # 6. Display results
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    print("=" * 70)

    return metrics