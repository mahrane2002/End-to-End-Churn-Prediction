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
from src.preprocessing.preprocessing import create_preprocessor


def _preprocess_fold(
    X_fold_train: pd.DataFrame,
    X_fold_val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess one cross-validation fold.

    The preprocessor is fitted only on the training fold.
    The validation fold is transformed using the fitted
    preprocessor.

    Parameters
    ----------
    X_fold_train : pd.DataFrame
        Training portion of the fold.

    X_fold_val : pd.DataFrame
        Validation portion of the fold.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Preprocessed training and validation data.
    """

    # ==============================================================
    # 1. Identify feature types from training fold only
    # ==============================================================

    numerical_features = X_fold_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_fold_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # ==============================================================
    # 2. Create preprocessing transformer
    # ==============================================================

    preprocessor = create_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    # ==============================================================
    # 3. Fit ONLY on training fold
    # ==============================================================

    X_fold_train_processed = preprocessor.fit_transform(
        X_fold_train
    )

    # ==============================================================
    # 4. Transform validation fold
    # ==============================================================

    X_fold_val_processed = preprocessor.transform(
        X_fold_val
    )

    # ==============================================================
    # 5. Recover transformed feature names
    # ==============================================================

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
    """Optuna objective function for XGBoost hyperparameter optimization.

    Cross-validation is performed using only the training data.

    For every fold:
    - preprocessing is fitted only on the training fold;
    - the validation fold is only transformed;
    - SMOTETomek is applied only to the training fold;
    - XGBoost is trained on the resampled training fold;
    - ROC-AUC is evaluated on the untouched validation fold.

    No feature selection is performed in this project.
    XGBoost uses the complete engineered feature set.

    Parameters
    ----------
    trial : optuna.Trial
        Current Optuna trial.

    X_train : pd.DataFrame
        Training features after feature engineering
        and before preprocessing.

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
        #
        # IMPORTANT:
        # The preprocessor is fitted ONLY on the training fold.
        # ----------------------------------------------------------

        (
            X_fold_train_processed,
            X_fold_val_processed,
        ) = _preprocess_fold(
            X_fold_train=X_fold_train,
            X_fold_val=X_fold_val,
        )

        # ----------------------------------------------------------
        # SMOTETomek
        #
        # IMPORTANT:
        # Applied ONLY to the training fold.
        # The validation fold remains untouched.
        # ----------------------------------------------------------

        sampler = SMOTETomek(
            random_state=RANDOM_STATE,
        )

        (
            X_fold_resampled,
            y_fold_resampled,
        ) = sampler.fit_resample(
            X_fold_train_processed,
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
            X_fold_val_processed
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
    # 4. Return mean CV ROC-AUC
    # ==============================================================

    return float(np.mean(fold_scores))


def tune_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
) -> tuple[dict[str, Any], float, optuna.Study]:
    """Optimize XGBoost hyperparameters using Optuna.

    The optimization is performed only on the training data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features after feature engineering
        and before preprocessing.

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
    # 1. Optuna sampler
    # ==============================================================

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE,
    )

    # ==============================================================
    # 2. Create study
    # ==============================================================

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="xgboost_churn_optimization",
    )

    # ==============================================================
    # 3. Run optimization
    # ==============================================================

    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # ==============================================================
    # 4. Return best result
    # ==============================================================

    return (
        study.best_params,
        study.best_value,
        study,
    )