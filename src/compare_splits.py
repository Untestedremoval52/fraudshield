import pandas as pd
from sklearn.metrics import average_precision_score
from src.features import build_dataset, target
from src.train import split_data, split_by_time, build_models, tune_threshold, evaluate
if __name__ == "__main__":
    df = build_dataset()
    rows = []
    for split_name, split_fn in [("random", split_data), ("time", split_by_time)]:
        X_train, X_val, X_test, y_train, y_val, y_test, held = split_fn(df)
        print(split_name, "fraud rate train/val/test:", round(y_train.mean(), 4), round(y_val.mean(), 4), round(y_test.mean(), 4))
        for name, model in build_models().items():
            model.fit(X_train, y_train)
            proba_val = model.predict_proba(X_val)[: , 1]
            threshold, _ = tune_threshold(y_val, proba_val)
            proba_test = model.predict_proba(X_test)[: , 1]
            scores = evaluate(y_test, proba_test, threshold)
            rows.append({"split" : split_name, "model" : name, "val_pr_auc" : round(average_precision_score(y_val, proba_val), 3), **{k : round(v, 3) for k, v in scores.items()}})
    table = pd.DataFrame(rows)
    print(table)
    table.to_csv("reports/split_comparison.csv", index = False)