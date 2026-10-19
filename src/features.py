import numpy as np
import pandas as pd
data_path = "data/transactions.csv"
warm_up_days = 30
app_channels = ["UPI", "Card_online", "Net_banking"]
target = "is_fraud"
categorical = ["channel", "merchant_category"]
numeric = ["amount", "log_amount", "amount_ratio", "hour", "is_night", "mins_since_login", "otp_fill_seconds", "failed_attempts", "active_call", "screen_share", "new_app_installed", "is_new_channel", "is_new_beneficiary", "is_new_device", "is_new_city", "ben_fan_in", "ben_is_merchant"]
def load_transactions(path):
    df = pd.read_csv(path, parse_dates = ["timestamp"])
    return df.sort_values("timestamp").reset_index(drop = True)
def add_features(df):
    df["is_new_channel"] = (df.groupby(["customer_id", "channel"]).cumcount() == 0).astype(int)
    df["is_new_beneficiary"] = (df.groupby(["customer_id", "beneficiary"]).cumcount() == 0).astype(int)
    df["is_new_city"] = (df.groupby(["customer_id", "city"]).cumcount() == 0).astype(int)
    df["is_new_device"] = ((df.groupby(["customer_id", "device_id"]).cumcount() == 0) & df["channel"].isin(app_channels)).astype(int)
    prev_mean = df.groupby("customer_id")["amount"].transform(lambda s : s.shift().expanding().mean())
    df["amount_ratio"] = (df["amount"] / prev_mean).fillna(1.0).round(3)
    df["log_amount"] = np.log1p(df["amount"]).round(4)
    df["hour"] = df["timestamp"].dt.hour
    df["is_night"] = df["hour"].between(0, 5).astype(int)
    df["ben_fan_in"] = df.groupby("beneficiary")["customer_id"].transform(lambda s : (~s.duplicated()).cumsum())
    return df
def build_dataset(path = data_path):
    df = add_features(load_transactions(path))
    start = df["timestamp"].min().normalize()
    df = df[df["timestamp"] >= start + pd.Timedelta(days = warm_up_days)].reset_index(drop = True)
    return df
if __name__ == "__main__":
    df = build_dataset()
    print(df.shape, df[target].sum())
    for f in ["is_new_channel", "is_new_beneficiary", "is_new_device", "is_new_city"]:
        print(f, df.groupby(target)[f].mean().round(3).to_dict())
    print(df.groupby(target)["amount_ratio"].median())
    print(df.groupby(target)["ben_fan_in"].median())