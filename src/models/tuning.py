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
    """
    Optuna objective function for XGBoost hyperparameter optimization.

    SMOTETomek is applied only to the training portion of each
    cross-validation fold. The validation fold remains untouched.

    Parameters
    ----------
    trial : optuna.Trial
        Current Optuna trial.

    X_train : pd.DataFrame
        Training features.

    y_train : pd.Series
        Training target.

    Returns
    -------
    float
        Mean cross-validation ROC-AUC.
    """

    # ------------------------------------------------------------------
    # 1. Hyperparameter search space
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 2. Stratified K-Fold cross-validation
    # ------------------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_scores = []

    # ------------------------------------------------------------------
    # 3. Cross-validation
    # ------------------------------------------------------------------

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_train, y_train),
        start=1,
    ):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]

        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        # --------------------------------------------------------------
        # SMOTETomek ONLY on the training fold
        # --------------------------------------------------------------

        sampler = SMOTETomek(
            random_state=RANDOM_STATE,
        )

        X_fold_resampled, y_fold_resampled = sampler.fit_resample(
            X_fold_train,
            y_fold_train,
        )

        # --------------------------------------------------------------
        # 4. Create XGBoost model
        # --------------------------------------------------------------

        model = XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        # --------------------------------------------------------------
        # 5. Train on resampled training fold
        # --------------------------------------------------------------

        model.fit(
            X_fold_resampled,
            y_fold_resampled,
        )

        # --------------------------------------------------------------
        # 6. Evaluate on untouched validation fold
        # --------------------------------------------------------------

        y_val_proba = model.predict_proba(
            X_fold_val
        )[:, 1]

        fold_auc = roc_auc_score(
            y_fold_val,
            y_val_proba,
        )

        fold_scores.append(fold_auc)

        # --------------------------------------------------------------
        # 7. Optuna pruning
        # --------------------------------------------------------------

        mean_score = float(np.mean(fold_scores))

        trial.report(
            mean_score,
            step=fold,
        )

        if trial.should_prune():
            raise optuna.TrialPruned()

    # ------------------------------------------------------------------
    # 8. Return mean CV ROC-AUC
    # ------------------------------------------------------------------

    return float(np.mean(fold_scores))


def tune_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
) -> tuple[dict[str, Any], float, optuna.Study]:
    """
    Optimize XGBoost hyperparameters using Optuna.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.

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

    # ------------------------------------------------------------------
    # 1. Create Optuna sampler
    # ------------------------------------------------------------------

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE,
    )

    # ------------------------------------------------------------------
    # 2. Create optimization study
    # ------------------------------------------------------------------

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="xgboost_churn_optimization",
    )

    # ------------------------------------------------------------------
    # 3. Run optimization
    # ------------------------------------------------------------------

    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # ------------------------------------------------------------------
    # 4. Return best result
    # ------------------------------------------------------------------

    return (
        study.best_params,
        study.best_value,
        study,
    )