import mlflow


print("=" * 70)
print("MLFLOW INFORMATION")
print("=" * 70)

print(f"Tracking URI: {mlflow.get_tracking_uri()}")

print("\n" + "=" * 70)
print("ALL LOGGED MODELS")
print("=" * 70)

models = mlflow.search_logged_models(
    output_format="list"
)

print(f"\nNumber of logged models: {len(models)}")

for model in models:
    print("\n" + "-" * 70)
    print(f"Model ID       : {model.model_id}")
    print(f"Name           : {model.name}")
    print(f"Source Run ID   : {model.source_run_id}")
    print(f"Status         : {model.status}")
    print(f"Artifact URI    : {model.artifact_location}")