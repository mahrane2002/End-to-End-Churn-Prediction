import numpy as np
import pandas as pd

from src.models.evaluate import evaluate_model


def test_evaluate_model():
    y_test = pd.Series([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8])

    result = evaluate_model(
        y_test,
        y_pred,
        y_proba,
    )

    assert isinstance(result, dict)

    assert "accuracy" in result
    assert "precision" in result
    assert "recall" in result
    assert "f1_score" in result
    assert "roc_auc" in result

    assert result["accuracy"] == 1.0
    assert result["f1_score"] == 1.0