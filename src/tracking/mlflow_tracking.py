import mlflow


EXPERIMENT_NAME = "churn_prediction"


def setup_mlflow() -> None:
    """Configure the MLflow experiment."""

    mlflow.set_experiment(EXPERIMENT_NAME)