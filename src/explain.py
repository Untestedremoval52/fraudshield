import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from src.features import build_dataset, categorical, numeric
from src.train import split_by_time
def make_explainer(model):
    return shap.TreeExplainer(model[-1])
def clean_names(model):
    return [n.split("__")[1] for n in model[0].get_feature_names_out()]
def explain_one(model, explainer, row, top = 3):
    xt = model[0].transform(row)
    vals = explainer.shap_values(xt)[0]
    s = pd.Series(vals, index = clean_names(model))
    s = s[s > 0].sort_values(ascending = False).head(top)
    return [(f, round(float(v), 2)) for f, v in s.items()]
if __name__ == "__main__":
    bundle = joblib.load("models/fraud_model.joblib")
    model = bundle["model"]
    df = build_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test, held = split_by_time(df)
    explainer = make_explainer(model)
    sample = model[0].transform(X_test.iloc[ : 2000])
    sv = explainer.shap_values(sample)
    importance = pd.Series(np.abs(sv).mean(axis = 0), index = clean_names(model)).sort_values(ascending = False)
    print(importance.head(10))
    importance.to_csv("reports/shap_importance.csv")
    shap.summary_plot(sv, sample, feature_names = clean_names(model), max_display = 12, show = False)
    plt.tight_layout()
    plt.savefig("reports/figures/shap_summary.png", dpi = 150)
    plt.close()
    p_test = model.predict_proba(X_test)[ : , 1]
    i = int(np.argmax(p_test))
    print("Most suspicious test payment:", round(float(p_test[i]), 3), explain_one(model, explainer, X_test.iloc[[i]]))
    X_held = held[categorical + numeric]
    p_held = model.predict_proba(X_held)[ : , 1]
    j = int(np.argmax(p_held))
    print("Most suspicious held payment:", round(float(p_held[j]), 3), explain_one(model, explainer, X_held.iloc[[j]]))