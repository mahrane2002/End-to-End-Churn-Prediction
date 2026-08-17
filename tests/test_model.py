"""Tests for preprocessing, feature selection, training, and artifact persistence."""

import tempfile
from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier

from src.config.config import TARGET_COLUMN
from src.data.data_ingestion import load_data
from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.preprocessing import preprocess_data
from src.preprocessing.feature_selection import select_features
from src.models.train import train_model
from src.models.predict import predict, predict_proba
from src.utils.artifact_manager import (
    save_model,
    load_model,
    save_preprocessor,
    load_preprocessor,
    save_selector,
    load_selector,
    save_metadata,
    load_metadata,
)


def test_preprocessing_and_selection():
    """Test that preprocess_data and select_features successfully transform data."""
    df = load_data().iloc[:100]  # Use a small subset
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_eng = engineer_features(X)
    assert "BalancePerAge" in X_eng.columns

    X_train_proc, X_test_proc, preprocessor = preprocess_data(X_eng, X_eng)
    assert not X_train_proc.empty
    assert X_train_proc.shape[1] > 0

    X_train_sel, X_test_sel, selector = select_features(
        X_train=X_train_proc,
        y_train=y,
        X_test=X_test_proc,
        k=5,
    )
    assert X_train_sel.shape[1] == 5
    assert X_test_sel.shape[1] == 5


def test_training_and_prediction():
    """Test train_model, predict, and predict_proba."""
    df = load_data().iloc[:100]
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_eng = engineer_features(X)
    X_proc, _, preprocessor = preprocess_data(X_eng, X_eng)
    X_sel, _, selector = select_features(X_proc, y, X_test=None, k=5)

    dummy_params = {
        "n_estimators": 5,
        "max_depth": 2,
        "learning_rate": 0.1,
    }

    model = train_model(X_sel, y, best_params=dummy_params)
    assert isinstance(model, XGBClassifier)

    preds = predict(model, X_sel)
    probs = predict_proba(model, X_sel)

    assert len(preds) == len(X_sel)
    assert len(probs) == len(X_sel)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_save_load_artifacts():
    """Test save and load functions of artifact_manager."""
    df = load_data().iloc[:50]
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_eng = engineer_features(X)
    X_proc, _, preprocessor = preprocess_data(X_eng, X_eng)
    X_sel, _, selector = select_features(X_proc, y, X_test=None, k=5)

    dummy_params = {
        "n_estimators": 2,
        "max_depth": 2,
        "learning_rate": 0.1,
    }
    model = train_model(X_sel, y, best_params=dummy_params)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        model_path = tmp_path / "model.pkl"
        prep_path = tmp_path / "preprocessor.pkl"
        sel_path = tmp_path / "selector.pkl"
        meta_path = tmp_path / "metadata.json"

        # Save
        save_model(model, path=model_path)
        save_preprocessor(preprocessor, path=prep_path)
        save_selector(selector, path=sel_path)

        meta = {"model_type": "XGBoost", "n_features": 5}
        save_metadata(meta, path=meta_path)

        # Load
        loaded_model = load_model(path=model_path)
        loaded_prep = load_preprocessor(path=prep_path)
        loaded_sel = load_selector(path=sel_path)
        loaded_meta = load_metadata(path=meta_path)

        assert isinstance(loaded_model, XGBClassifier)
        assert loaded_meta["n_features"] == 5
        assert loaded_prep.get_feature_names_out() is not None
        assert loaded_sel.get_feature_names_out() is not None
