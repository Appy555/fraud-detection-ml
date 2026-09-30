import os
from typing import Dict, Any, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time Machine Learning API for detecting fraudulent credit card transactions.",
    version="1.0.0"
)

# Robust base directory resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_model.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "model", "threshold.pkl")

# Load final trained pipeline (StandardScaler + LogisticRegression)
try:
    model = joblib.load(MODEL_PATH)
    threshold = float(joblib.load(THRESHOLD_PATH))
except Exception as e:
    # Fallback to local relative path if BASE_DIR resolution differs
    model = joblib.load("model/fraud_model.pkl")
    threshold = float(joblib.load("model/threshold.pkl"))

# Determine expected feature column order from fitted model
if hasattr(model, "feature_names_in_"):
    EXPECTED_COLUMNS = list(model.feature_names_in_)
elif hasattr(model.named_steps.get("standardscaler", None), "feature_names_in_"):
    EXPECTED_COLUMNS = list(model.named_steps["standardscaler"].feature_names_in_)
else:
    EXPECTED_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


class TransactionInput(BaseModel):
    Time: float = Field(..., description="Elapsed time in seconds since first transaction")
    V1: float = Field(..., description="PCA feature V1")
    V2: float = Field(..., description="PCA feature V2")
    V3: float = Field(..., description="PCA feature V3")
    V4: float = Field(..., description="PCA feature V4")
    V5: float = Field(..., description="PCA feature V5")
    V6: float = Field(..., description="PCA feature V6")
    V7: float = Field(..., description="PCA feature V7")
    V8: float = Field(..., description="PCA feature V8")
    V9: float = Field(..., description="PCA feature V9")
    V10: float = Field(..., description="PCA feature V10")
    V11: float = Field(..., description="PCA feature V11")
    V12: float = Field(..., description="PCA feature V12")
    V13: float = Field(..., description="PCA feature V13")
    V14: float = Field(..., description="PCA feature V14")
    V15: float = Field(..., description="PCA feature V15")
    V16: float = Field(..., description="PCA feature V16")
    V17: float = Field(..., description="PCA feature V17")
    V18: float = Field(..., description="PCA feature V18")
    V19: float = Field(..., description="PCA feature V19")
    V20: float = Field(..., description="PCA feature V20")
    V21: float = Field(..., description="PCA feature V21")
    V22: float = Field(..., description="PCA feature V22")
    V23: float = Field(..., description="PCA feature V23")
    V24: float = Field(..., description="PCA feature V24")
    V25: float = Field(..., description="PCA feature V25")
    V26: float = Field(..., description="PCA feature V26")
    V27: float = Field(..., description="PCA feature V27")
    V28: float = Field(..., description="PCA feature V28")
    Amount: float = Field(..., description="Transaction amount")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Time": 0.0,
                "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
                "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
                "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
                "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
                "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
                "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
                "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
                "Amount": 149.62
            }
        }
    }


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "Fraud Detection API",
        "docs_url": "/docs",
        "threshold": threshold
    }


@app.post("/predict")
def predict(data: TransactionInput):
    try:
        payload = data.model_dump()

        # Ensure all expected columns are present
        missing = [col for col in EXPECTED_COLUMNS if col not in payload]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required feature columns: {missing}"
            )

        # Construct DataFrame in exact feature order expected by scikit-learn
        input_data = pd.DataFrame([payload])[EXPECTED_COLUMNS]

        # Calculate fraud probability (Class 1)
        probability = float(model.predict_proba(input_data)[0][1])

        # Decision threshold comparison
        is_fraud = int(probability >= threshold)

        return {
            "prediction": is_fraud,
            "label": "FRAUD" if is_fraud == 1 else "LEGITIMATE",
            "fraud_probability": round(probability, 6),
            "threshold": threshold,
            "risk_level": "HIGH" if probability >= threshold else ("MEDIUM" if probability >= 0.50 else "LOW")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")