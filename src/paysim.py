import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from src.train import build_models, tune_threshold, evaluate
raw_path = "data/paysim.csv"
sample_path = "data/paysim_sample.csv"
sample_frac = 0.20
feature_sets = {"transaction_only" : ["amount", "log_amount", "hour", "dest_fan_in"], "with_balances" : ["amount", "log_amount", "hour", "dest_fan_in", "oldbalanceOrg", "newbalanceOrig", "drain_ratio"]}
def load_paysim():
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)
    chunks = []
    for chunk in pd.read_csv(raw_path, chunksize = 1000000):
        chunks.append(chunk[chunk["type"].isin(["TRANSFER", "CASH_OUT"])])
    df = pd.concat(chunks).sample(frac = sample_frac, random_state = 42).sort_values("step").reset_index(drop = True)
    df.to_csv(sample_path, index = False)
    return df
def add_paysim_features(df):
    df["log_amount"] = np.log1p(df["amount"])
    df["hour"] = df["step"] % 24
    df["drain_ratio"] = (df["amount"] / df["oldbalanceOrg"].replace(0, np.nan)).fillna(0).clip(0, 10)
    df["dest_fan_in"] = df.groupby("nameDest")["nameOrig"].transform(lambda s : (~s.duplicated()).cumsum())
    return df
if __name__ == "__main__":
    df = add_paysim_features(load_paysim())
    n = len(df)
    a, b = int(n * 0.70), int(n * 0.85)
    train, val, test = df.iloc[ : a], df.iloc[a : b], df.iloc[b : ]
    print(f"Size of train: {len(train)}, Size of value: {len(val)}, Size of test: {len(test)}, Fraud count: {df['isFraud'].sum()}")
    rows = []
    for set_name, num_cols in feature_sets.items():
        cols = ["type"] + num_cols
        for name, model in build_models(["type"], num_cols).items():
            model.fit(train[cols], train["isFraud"])
            proba_val = model.predict_proba(val[cols])[ : , 1]
            threshold, _ = tune_threshold(val["isFraud"], proba_val)
            proba_test = model.predict_proba(test[cols])[ : , 1]
            scores = evaluate(test["isFraud"], proba_test, threshold)
            rows.append({"features" : set_name, "model" : name, "val_pr_auc" : round(average_precision_score(val["isFraud"], proba_val), 3), **{k : round(v, 3) for k, v in scores.items()}})
            print(rows[-1])
    table = pd.DataFrame(rows)
    print(table)
    table.to_csv("reports/paysim_results.csv", index = False)