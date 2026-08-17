"""Hyperparameter tuning module for the Bank Customer Churn Prediction project."""

from typing import Any

import numpy as np
import optuna
import pandas as pd

from imblearn.combine import SMOTETomek
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.config.config import CV_FOLDS, RANDOM_STATE
from src.preprocessing.feature_selection import select_features
from src.preprocessing.preprocessing import create_preprocessor


def _preprocess_fold(
    X_fold_train: pd.DataFrame,
    X_fold_val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess one CV fold.

    The preprocessor is fitted only on the training fold.
    """

    numerical_features = X_fold_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_fold_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    preprocessor = create_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    # Fit ONLY on fold training data
    X_fold_train_processed = preprocessor.fit_transform(
        X_fold_train
    )

    # Transform validation data
    X_fold_val_processed = preprocessor.transform(
        X_fold_val
    )

    feature_names = preprocessor.get_feature_names_out()

    X_fold_train_processed = pd.DataFrame(
        X_fold_train_processed,
        columns=feature_names,
        index=X_fold_train.index,
    )

    X_fold_val_processed = pd.DataFrame(
        X_fold_val_processed,
        columns=feature_names,
        index=X_fold_val.index,
    )

    return (
        X_fold_train_processed,
        X_fold_val_processed,
    )


def objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> float:
    """Optuna objective using leakage-free cross-validation.

    For every fold:

    1. Split training and validation fold.
    2. Fit preprocessing on training fold only.
    3. Transform validation fold.
    4. Fit feature selection on training fold only.
    5. Transform validation fold using the fitted selector.
    6. Apply SMOTETomek only to the training fold.
    7. Train XGBoost.
    8. Evaluate ROC-AUC on untouched validation data.
    """

    # ==============================================================
    # 1. Hyperparameter search space
    # ==============================================================

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            100,
            500,
            step=50,
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            10,
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.3,
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0,
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0,
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            10,
        ),
        "gamma": trial.suggest_float(
            "gamma",
            0.0,
            5.0,
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-8,
            10.0,
            log=True,
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-8,
            10.0,
            log=True,
        ),
    }

    # ==============================================================
    # 2. Stratified K-Fold
    # ==============================================================

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_scores: list[float] = []

    # ==============================================================
    # 3. Cross-validation
    # ==============================================================

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_train, y_train),
        start=1,
    ):

        # ----------------------------------------------------------
        # Split fold
        # ----------------------------------------------------------

        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]

        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        # ----------------------------------------------------------
        # Preprocessing
        # ----------------------------------------------------------

        (
            X_fold_train_processed,
            X_fold_val_processed,
        ) = _preprocess_fold(
            X_fold_train=X_fold_train,
            X_fold_val=X_fold_val,
        )

        # ----------------------------------------------------------
        # Feature Selection
        #
        # IMPORTANT:
        # The selector is fitted ONLY on the fold training data.
        #
        # The validation fold is NEVER used to choose features.
        # ----------------------------------------------------------

        (
            X_fold_train_selected,
            X_fold_val_selected,
            _,
        ) = select_features(
            X_train=X_fold_train_processed,
            y_train=y_fold_train,
            X_test=X_fold_val_processed,
        )

        # ----------------------------------------------------------
        # SMOTETomek
        #
        # IMPORTANT:
        # Applied ONLY to the selected training fold.
        # ----------------------------------------------------------

        sampler = SMOTETomek(
            random_state=RANDOM_STATE,
        )

        (
            X_fold_resampled,
            y_fold_resampled,
        ) = sampler.fit_resample(
            X_fold_train_selected,
            y_fold_train,
        )

        # ----------------------------------------------------------
        # XGBoost
        # ----------------------------------------------------------

        model = XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            enable_categorical=False,
        )

        # ----------------------------------------------------------
        # Train
        # ----------------------------------------------------------

        model.fit(
            X_fold_resampled,
            y_fold_resampled,
        )

        # ----------------------------------------------------------
        # Validation
        # ----------------------------------------------------------

        y_val_proba = model.predict_proba(
            X_fold_val_selected
        )[:, 1]

        fold_auc = roc_auc_score(
            y_fold_val,
            y_val_proba,
        )

        fold_scores.append(fold_auc)

        # ----------------------------------------------------------
        # Optuna pruning
        # ----------------------------------------------------------

        mean_score = float(
            np.mean(fold_scores)
        )

        trial.report(
            mean_score,
            step=fold,
        )

        if trial.should_prune():
            raise optuna.TrialPruned()

    # ==============================================================
    # 4. Mean CV ROC-AUC
    # ==============================================================

    return float(
        np.mean(fold_scores)
    )


def tune_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
) -> tuple[
    dict[str, Any],
    float,
    optuna.Study,
]:
    """Optimize XGBoost hyperparameters using Optuna."""

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="xgboost_churn_optimization",
    )

    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    return (
        study.best_params,
        study.best_value,
        study,
    )