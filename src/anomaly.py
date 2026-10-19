import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from src.features import build_dataset, categorical, numeric
from src.train import split_by_time
alert_rates = [0.01, 0.02, 0.05]
cost_threshold = 0.18
def build_isolation_forest():
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown = "ignore"), categorical), ("num", "passthrough", numeric)])
    return make_pipeline(pre, IsolationForest(n_estimators = 300, random_state = 42))
if __name__ == "__main__":
    bundle = joblib.load("models/fraud_model.joblib")
    model, f1_threshold = bundle["model"], bundle["threshold"]
    df = build_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test, held = split_by_time(df)
    X_held = held[categorical + numeric]
    iso = build_isolation_forest()
    iso.fit(X_train[y_train == 0])
    score_val = -iso.decision_function(X_val)
    score_test = -iso.decision_function(X_test)
    score_held = -iso.decision_function(X_held)
    rows = []
    for rate in alert_rates:
        cut = np.quantile(score_val[y_val == 0], 1 - rate)
        rows.append({"detector" : f"Isolation Forest ({int(rate * 100)} % alerts)", "unseen_G6_caught" : round((score_held >= cut).mean(), 3), "known_fraud_caught" : round((score_test[y_test == 1] >= cut).mean(), 3)})
    p_held = model.predict_proba(X_held)[ : , 1]
    p_test = model.predict_proba(X_test)[ : , 1]
    for label, t in [("Supervised (F1 Threshold)", f1_threshold), ("Supervised (cost threshold)", cost_threshold)]:
            rows.append({"detector" : label, "unseen_G6_caught" : round((p_held >= t).mean(), 3), "known_fraud_caught" : round((p_test[y_test == 1] >= t).mean(), round(3))})
    cut2 = np.quantile(score_val[y_val == 0], 0.98)
    combined_held = (p_held >= cost_threshold) | (score_held >= cut2)
    combined_test = (p_test[y_test == 1] >= cost_threshold) | (score_test[y_test == 1] >= cut2)
    rows.append({"detector" : "Supervised (cost) OR IF 2%", "unseen_G6_caught" : round(combined_held.mean(), 3), "known_fraud_caught" : round(combined_test.mean(), 3)})
    table = pd.DataFrame(rows)
    print(table)
    table.to_csv("reports/unseen_fraud.csv", index = False)
    joblib.dump({"model" : iso, "cut" : cut2}, "models/anomaly_model.joblib", compress = 3)