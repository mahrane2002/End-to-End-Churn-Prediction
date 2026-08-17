"""Tests for the overall churn prediction pipeline and explainability."""

import tempfile
from pathlib import Path
import pandas as pd

from src.config.config import TARGET_COLUMN
from src.data.data_ingestion import load_data
from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.preprocessing import preprocess_data
from src.preprocessing.feature_selection import select_features
from src.models.train import train_model
from src.models.explain import (
    create_tree_explainer,
    explain_global,
    explain_customer,
)


def test_shap_explanations():
    """Test global and customer SHAP explanations."""
    df = load_data().iloc[:100]
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_eng = engineer_features(X)
    X_train_proc, X_test_proc, preprocessor = preprocess_data(X_eng, X_eng)
    X_train_sel, X_test_sel, selector = select_features(
        X_train=X_train_proc,
        y_train=y,
        X_test=X_test_proc,
        k=5,
    )

    dummy_params = {
        "n_estimators": 5,
        "max_depth": 2,
        "learning_rate": 0.1,
    }
    model = train_model(X_train_sel, y, best_params=dummy_params)

    # 1. Create TreeExplainer
    explainer = create_tree_explainer(
        model=model,
        background_data=X_train_sel,
        background_size=10,
    )
    assert explainer is not None

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 2. Test global explanation
        importance = explain_global(
            explainer=explainer,
            X_test=X_test_sel,
            output_dir=tmp_path,
        )
        assert isinstance(importance, pd.DataFrame)
        assert not importance.empty
        assert (tmp_path / "shap_feature_importance.png").exists()
        assert (tmp_path / "shap_feature_importance.csv").exists()

        # 3. Test customer explanation
        client_idx = X_test_sel.index[0]
        contributions = explain_customer(
            model=model,
            explainer=explainer,
            X_test=X_test_sel,
            client_index=client_idx,
            threshold=0.5,
            output_dir=tmp_path,
        )
        assert isinstance(contributions, pd.DataFrame)
        assert not contributions.empty
        assert (tmp_path / f"client_{client_idx}_waterfall.png").exists()
        assert (tmp_path / f"client_{client_idx}_shap.csv").exists()
