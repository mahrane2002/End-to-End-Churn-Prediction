import mlflow


MODEL_NAME = "churn_model"
RUN_ID = "03900817cca14b4f95f3f790ffd6a041"


def main() -> None:
    model_uri = f"runs:/{RUN_ID}/model"

    print("=" * 60)
    print("REGISTERING MODEL IN MLFLOW MODEL REGISTRY")
    print("=" * 60)

    print(f"Model name : {MODEL_NAME}")
    print(f"Run ID     : {RUN_ID}")
    print(f"Model URI  : {model_uri}")

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    print("\nModel registered successfully!")
    print(f"Model name    : {registered_model.name}")
    print(f"Model version : {registered_model.version}")


if __name__ == "__main__":
    main()