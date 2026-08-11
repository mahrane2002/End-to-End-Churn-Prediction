```python
"""Hyperparameter tuning module for the Bank Customer Churn Prediction project.

This module performs Bayesian hyperparameter optimization of the XGBoost
classifier using Optuna and Stratified K-Fold cross-validation.

The tuning strategy follows the production notebook:
- Optuna TPE sampler
- Stratified 5-Fold cross-validation
- ROC-AUC as optimization metric
- SMOTETomek applied independently inside each CV fold
- XGBoost classifier
"""

from typing import Any

import numpy as np
import optuna
import pandas as pd

from imblearn.combine import SMOTETomek
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.config.config import CV_FOLDS, RANDOM_STATE, SCORING_METRIC


def objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> float:
    """Optuna objective function.

    Each Optuna trial:
    1. Samples XGBoost hyperparameters.
    2. Performs Stratified K-Fold cross-validation.
    3. Applies SMOTETomek only on each training fold.
    4. Trains XGBoost on the resampled fold.
    5. Evaluates ROC-AUC on the untouched validation fold.
    6. Returns the mean CV ROC-AUC.

    Parameters
    ----------
    trial : optuna.Trial
        Current Optuna trial.

    X_train : pd.DataFrame
        Selected training features.

    y_train : pd.Series
        Training target.

    Returns
    -------
    float
        Mean cross-validation ROC-AUC.
    """

    # ------------------------------------------------------------------
    # XGBoost hyperparameter search space
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
    # Stratified cross-validation
    # ------------------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_scores = []

    # ------------------------------------------------------------------
    # Cross-validation
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
        # Apply SMOTETomek ONLY to the training fold.
        # The validation fold must remain untouched.
        # --------------------------------------------------------------

        sampler = SMOTETomek(
            random_state=RANDOM_STATE,
        )

        X_fold_resampled, y_fold_resampled = sampler.fit_resample(
            X_fold_train,
            y_fold_train,
        )

        # --------------------------------------------------------------
        # Train XGBoost
        # --------------------------------------------------------------

        model = XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(
            X_fold_resampled,
            y_fold_resampled,
        )

        # --------------------------------------------------------------
        # Evaluate on untouched validation fold
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
        # Optuna pruning
        # --------------------------------------------------------------

        mean_score = float(np.mean(fold_scores))

        trial.report(
            mean_score,
            step=fold,
        )

        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_scores))


def tune_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
) -> tuple[dict[str, Any], float, optuna.Study]:
    """Optimize XGBoost hyperparameters using Optuna.

    Parameters
    ----------
    X_train : pd.DataFrame
        Selected training features.

    y_train : pd.Series
        Training target.

    n_trials : int, default=50
        Number of Optuna trials.

    Returns
    -------
    tuple
        best_params:
            Best XGBoost hyperparameters.

        best_score:
            Best mean CV ROC-AUC.

        study:
            Completed Optuna study.
    """

    # ------------------------------------------------------------------
    # TPE sampler
    # ------------------------------------------------------------------

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE,
    )

    # ------------------------------------------------------------------
    # Create optimization study
    # ------------------------------------------------------------------

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="xgboost_churn_optimization",
    )

    # ------------------------------------------------------------------
    # Run optimization
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

    return (
        study.best_params,
        study.best_value,
        study,
    )


def build_tuned_model(
    best_params: dict[str, Any],
) -> XGBClassifier:
    """Build the final XGBoost model using optimized parameters.

    Parameters
    ----------
    best_params : dict
        Hyperparameters returned by Optuna.

    Returns
    -------
    XGBClassifier
        Configured XGBoost model.
    """

    model = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return model


if __name__ == "__main__":
    from src.data.data_ingestion import load_data
    from src.data.data_validation import validate_data
    from src.preprocessing.feature_engineering import engineer_features
    from src.preprocessing.feature_selection import select_features
    from src.preprocessing.preprocessing import preprocess_data

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------

    df = load_data()

    # ------------------------------------------------------------------
    # 2. Validate data
    # ------------------------------------------------------------------

    df = validate_data(df)

    # ------------------------------------------------------------------
    # 3. Feature engineering
    # ------------------------------------------------------------------

    df = engineer_features(df)

    # ------------------------------------------------------------------
    # 4. Preprocessing
    # ------------------------------------------------------------------

    (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
    ) = preprocess_data(df)

    # ------------------------------------------------------------------
    # 5. Feature selection
    # ------------------------------------------------------------------

    (
        X_train_selected,
        X_test_selected,
        selector,
    ) = select_features(
        X_train_processed,
        y_train,
        X_test_processed,
    )

    # ------------------------------------------------------------------
    # 6. Hyperparameter tuning
    # ------------------------------------------------------------------

    best_params, best_score, study = tune_model(
        X_train_selected,
        y_train,
        n_trials=50,
    )

    print("\n" + "=" * 70)
    print("OPTUNA HYPERPARAMETER TUNING")
    print("=" * 70)

    print(f"\nBest CV {SCORING_METRIC.upper()}: {best_score:.4f}")

    print("\nBest parameters:")

    for parameter, value in best_params.items():
        print(f"  {parameter}: {value}")

    print("\nNumber of completed trials:")
    print(len(study.trials))

    # ------------------------------------------------------------------
    # 7. Build final tuned model
    # ------------------------------------------------------------------

    tuned_model = build_tuned_model(best_params)

    print("\nTuned XGBoost model created successfully.")
```
