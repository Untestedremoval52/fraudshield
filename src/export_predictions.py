import joblib
import pandas as pd
from src.features import build_dataset
from src.train import split_by_time
if __name__ == "__main__":
    model = joblib.load("models/fraud_model.joblib")["model"]
    X_train, X_val, X_test, y_train, y_val, y_test, held = split_by_time(build_dataset())
    out = pd.DataFrame({"is_fraud" : y_test.to_numpy(), "fraud_probability" : model.predict_proba(X_test)[: , 1].round(4), "amount" : X_test["amount"].to_numpy()})
    out.to_csv("reports/test_predictions.csv", index = False)
    print(out.shape, out["is_fraud"].sum())