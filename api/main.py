from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import CustomerRequest
from src.utils.artifact_manager import load_inference_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading inference pipeline...")

    app.state.pipeline = load_inference_pipeline()

    print("Inference pipeline loaded successfully.")

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