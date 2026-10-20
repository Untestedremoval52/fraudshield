import os
from src.features import build_dataset, categorical, numeric
from src.train import split_by_time
inputs = [c for c in categorical + numeric if c not in ("log_amount", "is_night")]
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, held = split_by_time(build_dataset())
    sample = X_test[inputs].head(2000).copy()
    sample["is_fraud"] = y_test.head(2000).values
    os.makedirs("samples", exist_ok = True)
    sample.to_csv("samples/sample_batch.csv", index = False)
    print(sample.shape, sample["is_fraud"].sum())