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


def objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> float:
    """Optuna objective function for XGBoost hyperparameter optimization.

    The input features are already:
    - feature engineered
    - preprocessed
    - feature selected

    Cross-validation is performed only on the training data.

    For every fold:
    - the training fold is separated from the validation fold;
    - SMOTETomek is applied only to the training fold;
    - XGBoost is trained on the resampled training fold;
    - ROC-AUC is evaluated on the untouched validation fold.

    Parameters
    ----------
    trial : optuna.Trial
        Current Optuna trial.

    X_train : pd.DataFrame
        Training features after preprocessing and feature selection.

    y_train : pd.Series
        Training target.

    Returns
    -------
    float
        Mean cross-validation ROC-AUC.
    """

    # ==============================================================
    # 1. Hyperparameter search space
    # ==============================================================

    params: dict[str, Any] = {
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
        # Split current fold
        # ----------------------------------------------------------

        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]

        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        # ----------------------------------------------------------
        # SMOTETomek
        #
        # IMPORTANT:
        # SMOTETomek is applied ONLY to the fold training data.
        # The validation fold remains completely untouched.
        # ----------------------------------------------------------

        sampler = SMOTETomek(
            random_state=RANDOM_STATE,
        )

        X_fold_resampled, y_fold_resampled = sampler.fit_resample(
            X_fold_train,
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
            X_fold_val
        )[:, 1]

        fold_auc = roc_auc_score(
            y_fold_val,
            y_val_proba,
        )

        fold_scores.append(fold_auc)

        # ----------------------------------------------------------
        # Optuna pruning
        # ----------------------------------------------------------

        mean_score = float(np.mean(fold_scores))

        trial.report(
            mean_score,
            step=fold,
        )

        if trial.should_prune():
            raise optuna.TrialPruned()

    # ==============================================================
    # 4. Mean CV ROC-AUC
    # ==============================================================

    return float(np.mean(fold_scores))


def tune_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
) -> tuple[dict[str, Any], float, optuna.Study]:
    """Optimize XGBoost hyperparameters using Optuna.

    The input X_train must already be:
    - feature engineered
    - preprocessed
    - feature selected

    Only the training data is used during optimization.

    Parameters
    ----------
    X_train : pd.DataFrame
        Preprocessed and feature-selected training features.

    y_train : pd.Series
        Training target.

    n_trials : int, default=50
        Number of Optuna trials.

    Returns
    -------
    tuple
        best_params :
            Best XGBoost hyperparameters.

        best_score :
            Best mean cross-validation ROC-AUC.

        study :
            Completed Optuna study.
    """

    # ==============================================================
    # 1. Validate inputs
    # ==============================================================

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")

    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    if X_train.empty:
        raise ValueError("X_train is empty.")

    if y_train.empty:
        raise ValueError("y_train is empty.")

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train must contain the same number of samples."
        )

    # ==============================================================
    # 2. Optuna sampler
    # ==============================================================

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE,
    )

    # ==============================================================
    # 3. Create study
    # ==============================================================

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="xgboost_churn_optimization",
    )

    # ==============================================================
    # 4. Run optimization
    # ==============================================================

    study.optimize(
        lambda trial: objective(
            trial=trial,
            X_train=X_train,
            y_train=y_train,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # ==============================================================
    # 5. Return best result
    # ==============================================================

    return (
        study.best_params,
        study.best_value,
        study,
    )