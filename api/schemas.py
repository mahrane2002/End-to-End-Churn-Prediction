from typing import Any

from pydantic import BaseModel, Field


class CustomerRequest(BaseModel):
    CreditScore: int = Field(..., ge=0)

    Geography: str

    Gender: str

    Age: int = Field(..., ge=0)

    Tenure: int = Field(..., ge=0)

    Balance: float = Field(..., ge=0)

    NumOfProducts: int = Field(..., ge=1)

    HasCrCard: int = Field(..., ge=0, le=1)

    IsActiveMember: int = Field(..., ge=0, le=1)

    EstimatedSalary: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    churn: bool
    churn_probability: float


class FeatureExplanation(BaseModel):
    feature: str
    value: Any
    shap_value: float
    effect: str


class ExplanationResponse(BaseModel):
    prediction: int
    churn: bool
    churn_probability: float
    top_features: list[FeatureExplanation]
    explanation_method: str