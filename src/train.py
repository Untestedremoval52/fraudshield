from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from src.features import build_dataset, categorical, numeric, target
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
if __name__ == "__main__":
    df = build_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test, held = split_data(df)
    print(f"Size of X_train: {len(X_train)}, Size of X_val: {len(X_val)}, Size of X_test: {len(X_test)}, Size of held: {len(held)}, Fraud count of y_train: {y_train.sum()}, Fraud count of y_val: {y_val.sum()}, Fraud count of y_test: {y_test.sum()}")
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[: , 1]
        print(f"Name: {name}, PR-AUC: {average_precision_score(y_val, proba):.3f}, ROC AUC Score: {roc_auc_score(y_val, proba):.3f}")