"""Unit and integration tests for churn prediction API."""

from unittest.mock import patch
from fastapi.testclient import TestClient
import pytest
from src.app import app

client = TestClient(app)


def test_health_check():
    """Verify health endpoint returns 200 and expected payload."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "churn-predictor"}


def test_model_info():
    """Verify metadata endpoint returns 200."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data


def test_predict_high_risk():
    """Short tenure, month-to-month, high charge should yield high risk."""
    payload = {
        "customer_id": "CUST-001",
        "tenure_months": 2,
        "monthly_charges": 95.50,
        "total_charges": 191.0,
        "contract": "month-to-month",
        "has_tech_support": False,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "CUST-001"
    assert data["churn_prediction"] is True
    assert data["risk_level"] == "High"
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_medium_risk():
    """Tenure and contract factors combined to land in the Medium tier (0.40 - 0.69)."""
    payload = {
        "customer_id": "CUST-004",
        "tenure_months": 24,
        "monthly_charges": 60.0,
        "total_charges": 1440.0,
        "contract": "month-to-month",
        "has_tech_support": True,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_level"] == "Medium"
    assert 0.40 <= data["churn_probability"] < 0.70


def test_predict_low_risk():
    """Long tenure, two-year contract, low charge should yield low risk."""
    payload = {
        "customer_id": "CUST-002",
        "tenure_months": 60,
        "monthly_charges": 25.0,
        "total_charges": 1500.0,
        "contract": "two-year",
        "has_tech_support": True,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["churn_prediction"] is False
    assert data["risk_level"] == "Low"


@pytest.mark.parametrize(
    "invalid_payload,missing_key",
    [
        (
            {
                "tenure_months": 12,
                "monthly_charges": 50.0,
                "total_charges": 600.0,
                "contract": "one-year",
            },
            "customer_id",
        ),
        (
            {
                "customer_id": "CUST-003",
                "tenure_months": -5,
                "monthly_charges": 50.0,
                "total_charges": 600.0,
                "contract": "one-year",
            },
            "tenure_months",
        ),
    ],
)
def test_predict_validation_errors(invalid_payload, missing_key):
    """Pydantic should reject invalid inputs with a 422 Unprocessable Entity."""
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_internal_server_error():
    """Verify that unexpected inference exceptions return a 500 status code."""
    payload = {
        "customer_id": "CUST-005",
        "tenure_months": 12,
        "monthly_charges": 50.0,
        "total_charges": 600.0,
        "contract": "one-year",
        "has_tech_support": True,
    }
    with patch("src.app.compute_churn_risk", side_effect=RuntimeError("Model failure")):
        response = client.post("/predict", json=payload)
        assert response.status_code == 500
        assert "Inference failure" in response.json()["detail"]
        
