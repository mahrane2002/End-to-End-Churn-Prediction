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
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, Any]:
    """
    Evaluate model predictions on the test set.

    Parameters
    ----------
    y_test : pd.Series
        True target values.

    y_pred : np.ndarray
        Predicted class labels.

    y_proba : np.ndarray
        Predicted probabilities for the positive class.

    Returns
    -------
    dict[str, Any]
        Evaluation metrics and reports.
    """

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

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

    confusion = confusion_matrix(
        y_test,
        y_pred,
    )

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": confusion.tolist(),
        "classification_report": report,
    }

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion)

    print("\nClassification Report:")
    print(report)

    print("=" * 70)

    return metrics