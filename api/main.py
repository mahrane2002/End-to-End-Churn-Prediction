"""
FastAPI application for customer churn prediction.

Endpoints
---------
GET  /health
POST /predict
POST /explain
"""

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import (
    CustomerRequest,
    PredictionResponse,
    ExplanationResponse,
)

from src.models.explain import (
    create_tree_explainer,
    explain_prediction,
)

from src.utils.artifact_manager import (
    load_inference_pipeline,
    load_shap_background,
)


# ====================================================================
# Application lifespan
# ====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load ML artifacts and SHAP explainer when the API starts.

    The following objects are loaded only once:

    - Complete inference pipeline
    - SHAP background dataset
    - SHAP TreeExplainer

    They are then stored in app.state and reused
    by incoming requests.
    """

    print("=" * 70)
    print("Starting Customer Churn Prediction API")
    print("=" * 70)

    try:

        # ------------------------------------------------------------
        # 1. Load complete inference pipeline
        # ------------------------------------------------------------

        print("\nLoading inference pipeline...")

        pipeline = load_inference_pipeline()

        print(
            "Inference pipeline loaded successfully."
        )

        # ------------------------------------------------------------
        # 2. Load SHAP background data
        # ------------------------------------------------------------

        print("\nLoading SHAP background...")

        background = load_shap_background()

        print(
            "SHAP background loaded successfully."
        )

        # ------------------------------------------------------------
        # 3. Create SHAP TreeExplainer
        # ------------------------------------------------------------

        print("\nCreating SHAP explainer...")

        explainer = create_tree_explainer(
            model=pipeline.model,
            background_data=background,
        )

        print(
            "SHAP explainer created successfully."
        )

        # ------------------------------------------------------------
        # 4. Store dependencies in application state
        # ------------------------------------------------------------

        app.state.pipeline = pipeline
        app.state.shap_explainer = explainer

        print("\n" + "=" * 70)
        print("API startup completed successfully")
        print("=" * 70)

    except Exception as exc:

        print("\n" + "=" * 70)
        print("API startup failed")
        print("=" * 70)

        raise RuntimeError(
            "Failed to initialize ML inference dependencies."
        ) from exc

    yield

    # =================================================================
    # Shutdown
    # =================================================================

    print("\n" + "=" * 70)
    print("Shutting down API")
    print("=" * 70)


# ====================================================================
# FastAPI application
# ====================================================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production-oriented API for customer churn prediction "
        "using XGBoost and SHAP explainability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ====================================================================
# Health check
# ====================================================================

@app.get(
    "/health",
)
def health() -> dict[str, str]:
    """
    Check whether the API is running.
    """

    return {
        "status": "healthy",
    }


# ====================================================================
# Prediction endpoint
# ====================================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_customer(
    customer: CustomerRequest,
) -> PredictionResponse:
    """
    Predict customer churn.

    Pipeline:

        Raw customer
              ↓
        Feature Engineering
              ↓
        Preprocessing
              ↓
        Feature Selection
              ↓
        XGBoost
              ↓
        Churn probability
              ↓
        Prediction
    """

    try:

        # ------------------------------------------------------------
        # 1. Convert Pydantic model to dictionary
        # ------------------------------------------------------------

        customer_data = customer.model_dump()

        # ------------------------------------------------------------
        # 2. Convert dictionary to DataFrame
        # ------------------------------------------------------------

        customer_df = pd.DataFrame(
            [customer_data]
        )

        # ------------------------------------------------------------
        # 3. Retrieve inference pipeline
        # ------------------------------------------------------------

        pipeline = app.state.pipeline

        # ------------------------------------------------------------
        # 4. Generate probability
        # ------------------------------------------------------------

        churn_probability = float(
            pipeline.predict_proba(
                customer_df
            )[0]
        )

        # ------------------------------------------------------------
        # 5. Apply explicit classification threshold
        # ------------------------------------------------------------
        #
        # Project policy:
        # threshold = 0.5
        #
        # We explicitly calculate the prediction
        # from the probability so /predict and /explain
        # use exactly the same decision rule.
        # ------------------------------------------------------------

        prediction = int(
            churn_probability >= 0.5
        )

        # ------------------------------------------------------------
        # 6. Return response
        # ------------------------------------------------------------

        return PredictionResponse(
            prediction=prediction,
            churn=bool(prediction),
            churn_probability=churn_probability,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc


# ====================================================================
# Explain endpoint
# ====================================================================

@app.post(
    "/explain",
    response_model=ExplanationResponse,
)
def explain_customer_endpoint(
    customer: CustomerRequest,
) -> ExplanationResponse:
    """
    Predict customer churn and explain the prediction using SHAP.

    Pipeline:

        Raw customer
              ↓
        Feature Engineering
              ↓
        Preprocessing
              ↓
        Feature Selection
              ↓
        XGBoost
              ↓
        SHAP
              ↓
        Top feature contributions
    """

    try:

        # ------------------------------------------------------------
        # 1. Convert request to DataFrame
        # ------------------------------------------------------------

        customer_data = customer.model_dump()

        customer_df = pd.DataFrame(
            [customer_data]
        )

        # ------------------------------------------------------------
        # 2. Retrieve dependencies
        # ------------------------------------------------------------

        pipeline = app.state.pipeline

        explainer = app.state.shap_explainer

        # ------------------------------------------------------------
        # 3. Transform raw customer
        # ------------------------------------------------------------
        #
        # IMPORTANT:
        #
        # We do NOT use SMOTETomek here.
        #
        # Production inference:
        #
        # Raw
        #   ↓
        # Feature Engineering
        #   ↓
        # Preprocessing
        #   ↓
        # Feature Selection
        #   ↓
        # X_ready
        #
        # X_ready is exactly the feature space
        # expected by XGBoost and SHAP.
        # ------------------------------------------------------------

        X_ready = pipeline.transform(
            customer_df
        )

        # ------------------------------------------------------------
        # 4. Generate SHAP explanation
        # ------------------------------------------------------------

        top_features = explain_prediction(
            model=pipeline.model,
            explainer=explainer,
            X_ready=X_ready,
            original_data=customer_df,
            top_k=5,
            threshold=0.5,
        )

        # ------------------------------------------------------------
        # 5. Calculate probability
        # ------------------------------------------------------------

        churn_probability = float(
            pipeline.model.predict_proba(
                X_ready
            )[0, 1]
        )

        # ------------------------------------------------------------
        # 6. Apply same threshold as /predict
        # ------------------------------------------------------------

        prediction = int(
            churn_probability >= 0.5
        )

        # ------------------------------------------------------------
        # 7. Return explanation response
        # ------------------------------------------------------------

        return ExplanationResponse(
            prediction=prediction,
            churn=bool(prediction),
            churn_probability=churn_probability,
            top_features=top_features,
            explanation_method="SHAP",
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Explanation failed.",
        ) from exc