import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from src.features import build_dataset
from src.train import split_by_time, build_models
if __name__ == "__main__":
    df = build_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test, held = split_by_time(df)
    X_dev = pd.concat([X_train, X_val])
    y_dev = pd.concat([y_train, y_val])
    cv = TimeSeriesSplit(n_splits = 5)
    for i, (tr, te) in enumerate(cv.split(X_dev)):
        print(f"fold {i + 1}: train {len(tr)}, test {len(te)}, frauds in test: {int(y_dev.iloc[te].sum())}")
    rows = []
    for name, model in build_models().items():
        result = cross_validate(model, X_dev, y_dev, cv = cv, scoring = "average_precision")
        s = result["test_score"]
        rows.append({"model" : name, "folds" : np.round(s, 3).tolist(), "mean_pr_auc" : round(s.mean(), 3), "std" : round(s.std(), 3)})
    table = pd.DataFrame(rows).sort_values("mean_pr_auc", ascending = False)
    print(table)
    table.to_csv("reports/cv_selection.csv", index = False)