import numpy as np
import pandas as pd
random_state = 42
n_customers = 5000
n_transactions = 100000
n_days = 90
start_date = "2026-01-01"
fraud_rate = 0.018
channels = ["UPI", "Card_online", "Net_banking", "ATM", "AEPS"]
merchant_categories = ["grocery", "food", "travel", "electronics", "utilities", "fashion", "entertainment", "transfer"]
cities = ["Delhi", "Mumbai", "Bengaluru", "Kolkata", "Chennai", "Hyderabad", "Pune", "Jaipur", "Lucknow", "Patna"]
group_weights = {"G1" : 0.30, "G2" : 0.30, "G3" : 0.15, "G5" : 0.08, "G6" : 0.05, "G7" : 0.07, "G8" : 0.05}
rng = np.random.default_rng(random_state)
def make_customers(n):
    customer_id = [f"C{i:05d}" for i in range(n)]
    device_id = [f"D{i:05d}" for i in range(n)]
    account_age_days = rng.integers(30, 3650, n)
    typical_amount = rng.lognormal(mean = 7, sigma = 0.8, size = n).round(2)
    active_hour = rng.integers(8, 22, n)
    home_city = rng.choice(cities, n)
    usual_channels = [list(rng.choice(["UPI", "Card_online", "Net_banking", "ATM"], size = rng.integers(1, 4), replace = False, p = [0.5, 0.2, 0.15, 0.15])) for _ in range(n)]
    pool = [f"B{i:05d}" for i in range(20000)]
    regular_beneficiaries = [list(rng.choice(pool, size = 5, replace = False)) for _ in range(n)]
    return pd.DataFrame({"customer_id" : customer_id, "device_id" : device_id, "account_age_days" : account_age_days, "typical_amount" : typical_amount, "active_hour" : active_hour, "home_city" : home_city, "usual_channels" : usual_channels, "regular_beneficiaries" : regular_beneficiaries})
def make_genuine_transactions(customers, n):
    idx = rng.integers(0, len(customers), n)
    c = customers.iloc[idx].reset_index(drop = True)
    day = rng.integers(0, n_days, n)
    hour = np.clip(rng.normal(c["active_hour"], 3), 0, 23).astype(int)
    minute = rng.integers(0, 60, n)
    timestamp = pd.Timestamp(start_date) + pd.to_timedelta(day, unit = "D") + pd.to_timedelta(hour, unit = "h") + pd.to_timedelta(minute, unit = "m")
    amount = (c["typical_amount"] * rng.lognormal(0, 0.5, n)).round(2)
    channel = [rng.choice(ch) for ch in c["usual_channels"]]
    beneficiary = [rng.choice(b) if rng.random() < 0.9 else f"B{rng.integers(0, 20000):05d}" for b in c["regular_beneficiaries"]]
    return pd.DataFrame({"customer_id" : c["customer_id"], "timestamp" : timestamp, "amount" : amount, "channel" : channel, "beneficiary" : beneficiary, "device_id" : c["device_id"], "city" : c["home_city"], "merchant_category" : rng.choice(merchant_categories, n), "mins_since_login" : rng.uniform(1, 20, n).round(1), "otp_fill_seconds" : rng.uniform(5, 40, n).round(1), "failed_attempts" : rng.choice([0, 1, 2], n, p = [0.9, 0.08, 0.02]), "active_call" : 0, "screen_share" : 0, "new_app_installed" : 0, "is_fraud" : 0, "fraud_group" : "none"})
def add_genuine_noise(df):
    n_noise = int(len(df) * 0.03)
    noise_idx = rng.choice(df.index, size = n_noise, replace = False)
    travel, festive, new_phone = np.array_split(noise_idx, 3)
    df.loc[travel, "city"] = rng.choice(cities, len(travel))
    df.loc[festive, "amount"] *= 5
    df.loc[new_phone, "device_id"] = [f"DNEW{i:05d}" for i in range(len(new_phone))]
    return df
if __name__ == "__main__":
    customers = make_customers(n_customers)
    df = make_genuine_transactions(customers, n_transactions)
    df = add_genuine_noise(df)
    print("df shape: ", df.shape)
    print("customer shape: ", customers.shape)
    print(df["channel"].value_counts())
    print(df["amount"].describe())
    print("dates: ", df["timestamp"].min(), "to", df["timestamp"].max())
    print("fraud rows: ", df["is_fraud"].sum())