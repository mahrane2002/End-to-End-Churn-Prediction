from src.models.explain import create_tree_explainer
from src.utils.artifact_manager import (
    load_inference_pipeline,
    load_shap_background,
)


def load_dependencies():

    pipeline = load_inference_pipeline()

    background = load_shap_background()

    explainer = create_tree_explainer(
        model=pipeline.model,
        background_data=background,
    )

    return {
        "pipeline": pipeline,
        "explainer": explainer,
    }