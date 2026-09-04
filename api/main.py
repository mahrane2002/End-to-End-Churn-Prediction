
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all ML dependencies when the API starts.
    """

    print("Loading inference pipeline...")

    pipeline = load_inference_pipeline()

    print("Loading SHAP background...")

    background = load_shap_background()

    print("Creating SHAP explainer...")

    explainer = create_tree_explainer(
        model=pipeline.model,
        background_data=background,
    )

    # Store dependencies in application state
    app.state.pipeline = pipeline
    app.state.shap_explainer = explainer

    print("Inference pipeline loaded successfully.")
    print("SHAP explainer loaded successfully.")

    yield

    print("Shutting down API...")


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production-ready API for customer churn prediction "
        "with SHAP-based explanations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """
    Health check endpoint.
    """

    return {
        "status": "healthy",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_customer(
    customer: CustomerRequest,
):
    """
    Predict customer churn probability.
    """

    try:
        # --------------------------------------------------
        # 1. Convert Pydantic request to dictionary
        # --------------------------------------------------

        customer_data = customer.model_dump()

        # --------------------------------------------------
        # 2. Convert dictionary to DataFrame
        # --------------------------------------------------

        customer_df = pd.DataFrame(
            [customer_data]
        )

        # --------------------------------------------------
        # 3. Retrieve inference pipeline
        # --------------------------------------------------

        pipeline = app.state.pipeline

        # --------------------------------------------------
        # 4. Prediction
        # --------------------------------------------------

        prediction = pipeline.predict(
            customer_df
        )

        # --------------------------------------------------
        # 5. Churn probability
        # --------------------------------------------------

        probability = pipeline.predict_proba(
            customer_df
        )

        churn_probability = float(
            probability[0]
        )

        prediction_value = int(
            prediction[0]
        )

        # --------------------------------------------------
        # 6. Response
        # --------------------------------------------------

        return {
            "prediction": prediction_value,
            "churn": bool(prediction_value),
            "churn_probability": churn_probability,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from e


@app.post(
    "/explain",
    response_model=ExplanationResponse,
)
def explain_customer_endpoint(
    customer: CustomerRequest,
):
    """
    Predict customer churn and explain the prediction
    using SHAP.
    """

    try:
        # --------------------------------------------------
        # 1. Convert request to DataFrame
        # --------------------------------------------------

        customer_data = customer.model_dump()

        customer_df = pd.DataFrame(
            [customer_data]
        )

        # --------------------------------------------------
        # 2. Retrieve dependencies
        # --------------------------------------------------

        pipeline = app.state.pipeline
        explainer = app.state.shap_explainer

        # --------------------------------------------------
        # 3. Transform raw customer data
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
        # --------------------------------------------------

        X_ready = pipeline.transform(
            customer_df
        )

        # --------------------------------------------------
        # 4. SHAP explanation
        # --------------------------------------------------

        top_features = explain_prediction(
            model=pipeline.model,
            explainer=explainer,
            X_ready=X_ready,
            original_data=customer_df,
            top_k=5,
            threshold=0.5,
        )

        # --------------------------------------------------
        # 5. Prediction probability
        # --------------------------------------------------

        probability = float(
            pipeline.model.predict_proba(
                X_ready
            )[0, 1]
        )

        # --------------------------------------------------
        # 6. Prediction using threshold = 0.5
        # --------------------------------------------------

        prediction = int(
            probability >= 0.5
        )

        # --------------------------------------------------
        # 7. Response
        # --------------------------------------------------

        return {
            "prediction": prediction,
            "churn": bool(prediction),
            "churn_probability": probability,
            "top_features": top_features,
            "explanation_method": "SHAP",
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail="Explanation failed.",
        ) from e

