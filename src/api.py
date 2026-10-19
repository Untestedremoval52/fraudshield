import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from src.features import categorical, numeric
from src.explain import make_explainer, explain_one
cost_threshold = 0.18
fraud = joblib.load("models/fraud_model.joblib")
anomaly = joblib.load("models/anomaly_model.joblib")
model, block_threshold = fraud["model"], fraud["threshold"]
iso, iso_cut = anomaly["model"], anomaly["cut"]
explainer = make_explainer(model)
app = FastAPI(title = "FraudShield", description = "Scores a payment and decides: approve, step-up OTP or block.")
class Transaction(BaseModel):
    channel : str
    merchant_category : str
    amount : float
    amount_ratio : float
    hour : int
    mins_since_login : float
    otp_fill_seconds : float
    failed_attempts : int = 0
    active_call : int = 0
    screen_share : int = 0
    new_app_installed : int = 0
    is_new_channel : int = 0
    is_new_beneficiary : int = 0
    is_new_device : int = 0
    is_new_city : int = 0
    ben_fan_in : int = 0
    ben_is_merchant : int = 0
def to_row(tx):
    row = pd.DataFrame([tx.model_dump()])
    row["log_amount"] = np.log1p(row["amount"]).round(4)
    row["is_night"] = row["hour"].between(0, 5).astype(int)
    return row[categorical + numeric]
def decide(p, anomalous):
    if p >= block_threshold:
        return "Block"
    elif p >= cost_threshold or anomalous:
        return "Step_up_OTP"
    return "Approve"
@app.get("/health")
def health():
    return {"status" : "ok", "block_threshold" : round(block_threshold, 3), "otp_threshold" : cost_threshold}
@app.post("/score")
def score(tx : Transaction):
    row = to_row(tx)
    p = float(model.predict_proba(row)[0, 1])
    a = float(-iso.decision_function(row)[0])
    anomalous = a >= iso_cut
    return {"fraud_probability" : round(p, 3), "anomaly_score" : round(a, 3), "is_anomalous" : bool(anomalous), "decision" : decide(p, anomalous), "top_reasons" : explain_one(model, explainer, row)}