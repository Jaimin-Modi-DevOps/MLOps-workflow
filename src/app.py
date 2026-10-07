"""FastAPI application for Customer Churn Risk Prediction."""

from enum import Enum

import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

app = FastAPI(
    title="Customer Churn Prediction Service",
    description="MLOps Inference API with automated testing and CI/CD validation.",
    version="1.0.0",
)


class ContractType(str, Enum):
    """Allowed subscription contract types."""
    MONTH_TO_MONTH = "month-to-month"
    ONE_YEAR = "one-year"
    TWO_YEAR = "two-year"


class CustomerData(BaseModel):
    """Input payload schema for customer churn evaluation."""
    customer_id: str = Field(..., examples=["CUST-9821"])
    tenure_months: int = Field(..., ge=0, le=120, description="Tenure in months")
    monthly_charges: float = Field(..., gt=0.0, description="Monthly charges in USD")
    total_charges: float = Field(..., ge=0.0, description="Lifetime total charges")
    contract: ContractType
    has_tech_support: bool = Field(default=False)


class PredictionResponse(BaseModel):
    """Output schema for model prediction results."""
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    risk_level: str


def compute_churn_risk(data: CustomerData) -> dict[str, float | bool | str]:
    """Heuristic / inference mock simulating model scoring."""
    # Base risk factor
    score = 0.20

    if data.contract == ContractType.MONTH_TO_MONTH:
        score += 0.35
    elif data.contract == ContractType.TWO_YEAR:
        score -= 0.15

    if data.tenure_months < 12:
        score += 0.25
    elif data.tenure_months > 48:
        score -= 0.20

    if data.monthly_charges > 80.0:
        score += 0.15

    if not data.has_tech_support:
        score += 0.10

    # Clamp probability to [0.01, 0.99]
    probability = max(0.01, min(0.99, round(score, 4)))
    will_churn = probability >= 0.50

    if probability >= 0.70:
        risk_tier = "High"
    elif probability >= 0.40:
        risk_tier = "Medium"
    else:
        risk_tier = "Low"

    return {
        "churn_probability": probability,
        "churn_prediction": will_churn,
        "risk_level": risk_tier,
    }


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict[str, str]:
    """Liveness probe endpoint."""
    return {"status": "healthy", "service": "churn-predictor"}


@app.get("/model-info", status_code=status.HTTP_200_OK)
def model_info() -> dict[str, str]:
    """Model governance metadata endpoint."""
    return {
        "model_name": "churn_risk_classifier",
        "model_version": "v1.2.0",
        "framework": "scikit-learn / rule-engine",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
def predict(customer: CustomerData):
    """Predict customer churn risk based on account features."""
    try:
        result = compute_churn_risk(customer)
        return PredictionResponse(
            customer_id=customer.customer_id,
            churn_probability=result["churn_probability"],
            churn_prediction=result["churn_prediction"],
            risk_level=result["risk_level"],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure: {exc!s}",
        ) from exc


if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=False)
