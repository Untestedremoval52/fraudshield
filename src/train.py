from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from src.features import build_dataset, categorical, numeric, target
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, precision_score,recall_score, accuracy_score, f1_score, confusion_matrix, PrecisionRecallDisplay, ConfusionMatrixDisplay
random_state = 42
held_out_group = "G6"
def split_data(df):
    held = df[df["fraud_group"] == held_out_group]
    rest = df[df["fraud_group"] != held_out_group]
    X = rest[categorical + numeric]
    y = rest[target]
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size = 0.30, stratify = y, random_state = random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size = 0.50, stratify = y_tmp, random_state = random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test, held
def make_preprocessor(scale):
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown = "ignore"), categorical), ("num", StandardScaler() if scale else "passthrough", numeric)])
def build_models():
    return {"Logistic Regression" : make_pipeline(make_preprocessor(True), LogisticRegression(class_weight = "balanced", max_iter = 2000)), "Random Forest" : make_pipeline(make_preprocessor(False), RandomForestClassifier(n_estimators = 300, class_weight = "balanced", min_samples_leaf = 2, n_jobs = 1, random_state = random_state)), "Gradient Boosting" : make_pipeline(make_preprocessor(False), HistGradientBoostingClassifier(class_weight = "balanced", random_state = random_state))}
def tune_threshold(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = np.argmax(f1[:-1])
    return thresholds[best], f1[best]
def evaluate(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {"pr_auc" : average_precision_score(y_true, proba), "precision" : precision_score(y_true, pred), "recall" : recall_score(y_true, pred), "f1_score" : f1_score(y_true, pred)}
if __name__ == "__main__":
    df = build_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test, held = split_data(df)
    results, fitted = [], {}
    print(f"Size of X_train: {len(X_train)}, Size of X_val: {len(X_val)}, Size of X_test: {len(X_test)}, Size of held: {len(held)}, Fraud count of y_train: {y_train.sum()}, Fraud count of y_val: {y_val.sum()}, Fraud count of y_test: {y_test.sum()}")
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        proba_val = model.predict_proba(X_val)[: , 1]
        threshold, _ = tune_threshold(y_val, proba_val)
        scores = evaluate(y_val, proba_val, threshold)
        results.append({"model" : name, "threshold" : round(threshold, 3), **{k : round(v, 3) for k, v in scores.items()}})
        fitted[name] = (model, threshold, proba_val)
    table = pd.DataFrame(results).sort_values("pr_auc", ascending = False)
    print(table)
    os.makedirs("reports/figures", exist_ok = True)
    table.to_csv("reports/model_comparison.csv", index = False)
    best_name = table.iloc[0]["model"]
    best_model, best_threshold, _ = fitted[best_name]
    proba_test = best_model.predict_proba(X_test)[: , 1]
    test_scores = evaluate(y_test, proba_test, best_threshold)
    print(f"Best name: {best_name}, Test scores: {test_scores}")
    cm = confusion_matrix(y_test, (proba_test >= best_threshold).astype(int))
    print(f"Confusion matrix: {cm}")
    os.makedirs("models", exist_ok = True)
    joblib.dump({"model" : best_model, "threshold" : best_threshold, "name" : best_name}, "models/fraud_model.joblib", compress = 3)
    fig, ax = plt.subplots(figsize = (7, 6))
    for name, (_, _, proba_val) in fitted.items():
        PrecisionRecallDisplay.from_predictions(y_val, proba_val, name = name, ax = ax)
    ax.set_title("Precision-recall curves (validation)")
    plt.savefig("reports/figures/pr_curves.png", dpi = 150)
    plt.close()
    ConfusionMatrixDisplay(cm, display_labels = ["Genuine", "Fraud"]).plot(cmap = "Blues")
    plt.title(f"{best_name} - test set")
    plt.savefig("reports/figures/confusion_matrix.png", dpi = 150)
    plt.close()