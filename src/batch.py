import numpy as np
import pandas as pd
from src.api import model, iso, iso_cut, explainer, cost_threshold, block_threshold, Transaction
from src.features import categorical, numeric, add_derived
from src.explain import clean_names
from src.scenarios import reason_text
required = [n for n, f in Transaction.model_fields.items() if f.is_required()]
optional = [n for n, f in Transaction.model_fields.items() if not f.is_required()]
def prepare(df):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    df = df.copy()
    for c in optional:
        if c not in df.columns:
            df[c] = 0
    return add_derived(df)
def score_batch(df):
    df = prepare(df)
    X = df[categorical + numeric]
    p = model.predict_proba(X)[: , 1]
    anomalous = -iso.decision_function(X) >= iso_cut
    decision = np.select([p >= block_threshold, (p >= cost_threshold) | anomalous], ["Block", "Step_up_OTP"], default = "Approve")
    sv = explainer.shap_values(model[0].transform(X))
    top = np.array(clean_names(model))[sv.argmax(axis = 1)]
    out = df.copy()
    out["fraud_probability"] = p.round(3)
    out["is_anomalous"] = anomalous
    out["decision"] = decision
    out["top_reason"] = [reason_text(f, r) if d != "Approve" else "" for f, r, d in zip(top, df.to_dict("records"), decision)]
    return out