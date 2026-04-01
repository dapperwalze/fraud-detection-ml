from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Fraud Detection API")

model = joblib.load("fraud_model_xgb.pkl")
scaler = joblib.load("scaler.pkl")

class Transaction(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(transaction: Transaction):
    data = dict(transaction)
    amount = data.pop("Amount")
    amount_scaled = scaler.transform([[amount]])[0][0]
    features = list(data.values()) + [amount_scaled]
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0][1]
    return {
        "is_fraud": bool(prediction),
        "fraud_probability": round(float(probability), 4),
        "risk_level": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW"
    }
