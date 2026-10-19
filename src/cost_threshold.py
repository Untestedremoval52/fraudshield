import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.features import build_dataset
from src.train import split_by_time
false_alarm_costs = [50, 200, 1000]
thresholds = np.linspace(0.01, 0.99, 99)
def total_cost(y, proba, amounts, threshold, fa_cost):
    pred = proba >= threshold
    missed = (y == 1) & ~pred
    false_alarm = (y == 0) & pred
    cost = amounts[missed].sum() + fa_cost * false_alarm.sum()
    return cost, int(missed.sum()), int(false_alarm.sum())
def best_cost_threshold(y, proba, amounts, fa_cost):
    costs = [total_cost(y, proba, amounts, t, fa_cost)[0] for t in thresholds]
    return thresholds[int(np.argmin(costs))], costs
if __name__ == "__main__":
    bundle = joblib.load("models/fraud_model.joblib")
    model, f1_threshold = bundle["model"], bundle["threshold"]
    df = build_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test, held = split_by_time(df)
    proba_val = model.predict_proba(X_val)[ : , 1]
    proba_test = model.predict_proba(X_test)[ : , 1]
    y_val_arr, y_test_arr = y_val.to_numpy(), y_test.to_numpy()
    amt_val, amt_test = X_val["amount"].to_numpy(), X_test["amount"].to_numpy()
    rows = []
    for fa_cost in false_alarm_costs:
        cost_threshold, _ = best_cost_threshold(y_val_arr, proba_val, amt_val, fa_cost)
        for label, t in [("f1", f1_threshold), ("cost", cost_threshold)]:
            cost, missed, fas = total_cost(y_test_arr, proba_test, amt_test, t, fa_cost)
            rows.append({"false_alarm_cost" : fa_cost, "threshold_type" : label, "threshold" : round(t, 3), "fraud_missed" : missed, "false_alarms" : fas, "total_cost" : round(cost)})
    table = pd.DataFrame(rows)
    print(table)
    table.to_csv("reports/cost_analysis.csv", index = False)
    _, costs = best_cost_threshold(y_val_arr, proba_val, amt_val, 200)
    plt.figure(figsize = (8, 5))
    plt.plot(thresholds, costs)
    plt.axvline(f1_threshold, linestyle = '--', color = "red", label = "F1 threshold")
    plt.xlabel("threshold")
    plt.ylabel("total cost (Rs)")
    plt.title("Total cost vs threshold (false alarm = Rs 200)")
    plt.legend()
    plt.savefig("reports/figures/cost_curve.png", dpi = 150)
    plt.close()