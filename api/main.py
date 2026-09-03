from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import CustomerRequest,ExplanationResponse
from models.explain import create_tree_explainer
from src.utils.artifact_manager import load_inference_pipeline, load_shap_background
from src.models.explain import explain_prediction

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading inference pipeline...")

    pipeline = load_inference_pipeline()

    print("Loading SHAP background...")

    background = load_shap_background()

    print("Creating SHAP explainer...")

    explainer = create_tree_explainer(
        model=pipeline.model,
        background_data=background,
    )

    app.state.pipeline = pipeline
    app.state.shap_explainer = explainer

    print("Inference pipeline loaded successfully.")
    print("SHAP explainer loaded successfully.")

    yield

    print("Shutting down API...")


app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for customer churn prediction",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict_customer(customer: CustomerRequest):

    try:

        # Convert Pydantic object to dictionary
        customer_data = customer.model_dump()

        # Convert dictionary to DataFrame
        customer_df = pd.DataFrame(
            [customer_data]
        )

        # Get inference pipeline
        pipeline = app.state.pipeline

        # Prediction
        prediction = pipeline.predict(
            customer_df
        )

        # Probability
        probability = pipeline.predict_proba(
            customer_df
        )

        return {
            "prediction": int(prediction[0]),
            "churn_probability": float(
                probability[0]
            ),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
@app.post(
    "/explain",
    response_model=ExplanationResponse,
)
def explain_customer_endpoint(
    customer: CustomerRequest,
):

    try:

        customer_data = customer.model_dump()

        customer_df = pd.DataFrame(
            [customer_data]
        )

        pipeline = app.state.pipeline
        explainer = app.state.shap_explainer

        # Raw → engineered → preprocessed → selected
        X_ready = pipeline.transform(
            customer_df
        )

        # SHAP explanation
        top_features = explain_prediction(
            model=pipeline.model,
            explainer=explainer,
            X_ready=X_ready,
            original_data=customer_df,
            top_k=5,
            threshold=0.5,
        )

        probability = float(
            pipeline.model.predict_proba(
                X_ready
            )[0, 1]
        )

        prediction = int(
            probability >= 0.5
        )

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
            detail=str(e),
        )