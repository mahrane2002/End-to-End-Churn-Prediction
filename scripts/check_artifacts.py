"""Validate saved ML artifacts."""

from pathlib import Path
import sys

import joblib

from src.config.config import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    SELECTOR_PATH,
    METADATA_PATH,
)
from src.utils.artifact_manager import (
    load_model,
    load_preprocessor,
    load_selector,
    load_metadata,
)


def main() -> None:
    print("=" * 70)
    print("ARTIFACT VALIDATION")
    print("=" * 70)

    artifact_paths = {
        "model.pkl": MODEL_PATH,
        "preprocessor.pkl": PREPROCESSOR_PATH,
        "selector.pkl": SELECTOR_PATH,
        "metadata.json": METADATA_PATH,
    }

    # --------------------------------------------------------------
    # 1. Check files
    # --------------------------------------------------------------

    print("\n1. Checking artifact files...")

    for name, path in artifact_paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Missing artifact: {path}"
            )

        print(f"[OK] {name}: {path}")

    # --------------------------------------------------------------
    # 2. Load artifacts
    # --------------------------------------------------------------

    print("\n2. Loading artifacts...")

    model = load_model()
    preprocessor = load_preprocessor()
    selector = load_selector()
    metadata = load_metadata()

    print("[OK] model.pkl loaded")
    print("[OK] preprocessor.pkl loaded")
    print("[OK] selector.pkl loaded")
    print("[OK] metadata.json loaded")

    # --------------------------------------------------------------
    # 3. Validate model
    # --------------------------------------------------------------

    print("\n3. Model information")

    print(
        f"Model type: {type(model).__name__}"
    )

    if not hasattr(model, "predict"):
        raise TypeError(
            "Loaded model does not implement predict()."
        )

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Loaded model does not implement predict_proba()."
        )

    print("[OK] Model supports predict()")
    print("[OK] Model supports predict_proba()")

    # --------------------------------------------------------------
    # 4. Validate preprocessor
    # --------------------------------------------------------------

    print("\n4. Preprocessor information")

    if not hasattr(
        preprocessor,
        "transform",
    ):
        raise TypeError(
            "Loaded preprocessor does not implement transform()."
        )

    preprocessor_features = (
        list(
            preprocessor.get_feature_names_out()
        )
    )

    print(
        f"Preprocessed features: "
        f"{len(preprocessor_features)}"
    )

    print("[OK] Preprocessor is fitted")

    # --------------------------------------------------------------
    # 5. Validate selector
    # --------------------------------------------------------------

    print("\n5. Selector information")

    if not hasattr(
        selector,
        "transform",
    ):
        raise TypeError(
            "Loaded selector does not implement transform()."
        )

    selector_input_features = list(
        selector.feature_names_in_
    )

    selected_features = list(
        selector.get_feature_names_out(
            selector.feature_names_in_
        )
    )

    print(
        f"Selector input features: "
        f"{len(selector_input_features)}"
    )

    print(
        f"Selected features: "
        f"{len(selected_features)}"
    )

    # --------------------------------------------------------------
    # 6. Check preprocessor -> selector compatibility
    # --------------------------------------------------------------

    print("\n6. Checking preprocessor/selector compatibility...")

    if preprocessor_features != selector_input_features:
        missing = sorted(
            set(selector_input_features)
            - set(preprocessor_features)
        )

        unexpected = sorted(
            set(preprocessor_features)
            - set(selector_input_features)
        )

        raise ValueError(
            "Preprocessor and selector are incompatible.\n"
            f"Missing features: {missing}\n"
            f"Unexpected features: {unexpected}"
        )

    print(
        "[OK] Preprocessor and selector "
        "have identical feature names."
    )

    # --------------------------------------------------------------
    # 7. Validate metadata
    # --------------------------------------------------------------

    print("\n7. Metadata")

    required_metadata = [
        "model_type",
        "target",
        "selected_features",
        "n_features",
        "training_date",
    ]

    for key in required_metadata:
        if key not in metadata:
            raise ValueError(
                f"Missing metadata field: {key}"
            )

    metadata_selected_features = list(
        metadata["selected_features"]
    )

    if metadata_selected_features != selected_features:
        raise ValueError(
            "Metadata selected_features do not match "
            "selector selected features."
        )

    if int(metadata["n_features"]) != len(
        selected_features
    ):
        raise ValueError(
            "Metadata n_features does not match "
            "the selector."
        )

    print("[OK] Metadata is consistent")

    # --------------------------------------------------------------
    # 8. Final summary
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ALL ARTIFACT CHECKS PASSED")
    print("=" * 70)

    print("\nArtifact summary:")
    print(
        f"  Model              : {type(model).__name__}"
    )
    print(
        f"  Preprocessor input : "
        f"{len(preprocessor.feature_names_in_)} raw features"
    )
    print(
        f"  Preprocessor output: "
        f"{len(preprocessor_features)} features"
    )
    print(
        f"  Selector output    : "
        f"{len(selected_features)} features"
    )
    print(
        f"  Target             : "
        f"{metadata['target']}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\n[ERROR] Artifact validation failed:\n{exc}"
        )
        sys.exit(1)