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
mules = [f"M{i:03d}" for i in range(100)]
atm_terminals = [f"ATM{i:03d}" for i in range(500)]
aeps_terminals = [f"AEPS{i:02d}" for i in range(15)]
merchants = [f"MER{i:03d}" for i in range(300)]
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
def fix_atm_rows(df):
    atm_rows = df.index[df["channel"] == "ATM"]
    df.loc[atm_rows, "device_id"] = rng.choice(atm_terminals, len(atm_rows))
    df.loc[atm_rows, ["mins_since_login", "otp_fill_seconds"]] = -1
    return df
def pick_rows(df, k):
    return rng.choice(df.index[df["is_fraud"] == 0], k, replace = False)
def mark_fraud(df, idx, group):
    df.loc[idx, "is_fraud"] = 1
    df.loc[idx, "fraud_group"] = group
    df.loc[idx, "beneficiary"] = rng.choice(mules, len(idx))
def fraud_g1(df, k):
    idx = pick_rows(df, k)
    mark_fraud(df, idx, "G1")
    df.loc[idx, "device_id"] = [f"DF{i:05d}" for i in range(k)]
    df.loc[idx, "city"] = rng.choice(cities, k)
    df.loc[idx, "mins_since_login"] = rng.uniform(0.2, 3, k).round(1)
    df.loc[idx, "amount"] = (df.loc[idx, "amount"] * rng.uniform(2, 6, k)).round(2)
def fraud_g2(df, k):
    n_victims = k // 6
    victims = rng.choice(df["customer_id"].unique(), n_victims, replace = False)
    for v in victims:
        rows = df[(df["customer_id"] == v) & (df["is_fraud"] == 0)].sort_values("timestamp").index[-6:]
        if len(rows) < 6:
            continue
        base = df.loc[rows[0], "amount"]
        df.loc[rows, "amount"] = (base * 1.6 ** np.arange(6)).round(2)
        df.loc[rows, "is_fraud"] = 1
        df.loc[rows, "fraud_group"] = "G2"
        df.loc[rows, "beneficiary"] = rng.choice(mules)
def fraud_g3(df, k):
    idx = pick_rows(df, k)
    mark_fraud(df, idx, "G3")
    df.loc[idx, "active_call"] = 1
    df.loc[idx, "amount"] = (df.loc[idx, "amount"] * rng.uniform(1.5, 4, k)).round(2)
def fraud_g5(df, k):
    idx = pick_rows(df, k)
    mark_fraud(df, idx, "G5")
    df.loc[idx, "active_call"] = 1
    df.loc[idx, "screen_share"] = 1
    df.loc[idx, "amount"] = (df.loc[idx, "amount"] * rng.uniform(2, 5, k)).round(2)
def fraud_g6(df, k):
    idx = pick_rows(df, k)
    mark_fraud(df, idx , "G6")
    df.loc[idx, "otp_fill_seconds"] = rng.uniform(0.2, 1.5, k).round(1)
    df.loc[idx, "new_app_installed"] = 1
    df.loc[idx, "timestamp"] = df.loc[idx, "timestamp"].dt.normalize() + pd.to_timedelta(rng.integers(1, 5, k), unit = "h") + pd.to_timedelta(rng.integers(0, 60, k), unit = "m")
def fraud_g7(df, k):
    idx = pick_rows(df, k)
    mark_fraud(df, idx, "G7")
    df.loc[idx, "channel"] = "AEPS"
    df.loc[idx, "device_id"] = rng.choice(aeps_terminals, k)
    df.loc[idx, "mins_since_login"] = -1
    df.loc[idx, "otp_fill_seconds"] = -1
def fraud_g8(df, k):
    idx = pick_rows(df, k)
    mark_fraud(df, idx, "G8")
    df.loc[idx, "channel"] = "ATM"
    df.loc[idx, "device_id"] = rng.choice(atm_terminals, k)
    df.loc[idx, "city"] = rng.choice(cities, k)
    df.loc[idx, "amount"] = rng.choice([5000, 10000, 20000], k)
    df.loc[idx, "mins_since_login"] = -1
    df.loc[idx, "otp_fill_seconds"] = -1
def inject_fraud(df):
    df = fix_atm_rows(df)
    n_fraud = int(n_transactions * fraud_rate)
    counts = {g : int(round(w * n_fraud)) for g, w in group_weights.items()}
    fraud_g1(df, counts["G1"])
    fraud_g2(df, counts["G2"])
    fraud_g3(df, counts["G3"])
    fraud_g5(df, counts["G5"])
    fraud_g6(df, counts["G6"])
    fraud_g7(df, counts["G7"])
    fraud_g8(df, counts["G8"])
    return df
def add_genuine_overlap(df):
    n = len(df)
    to_merchant = rng.random(n) < 0.30
    df.loc[to_merchant, "beneficiary"] = rng.choice(merchants, to_merchant.sum())
    df.loc[rng.random(n) < 0.05, "active_call"] = 1
    df.loc[rng.random(n) < 0.005, "screen_share"] = 1
    df.loc[rng.random(n) < 0.03, "new_app_installed"] = 1
    autofill = rng.random(n) < 0.20
    df.loc[autofill, "otp_fill_seconds"] = rng.uniform(0.3, 3, autofill.sum()).round(1)
    quick = rng.random(n) < 0.25
    df.loc[quick, "mins_since_login"] = rng.uniform(0.2, 3, quick.sum()).round(1)
    return df
def weaken_fraud(df):
    idx = df.index[df["is_fraud"] == 1]
    one_off = idx[rng.random(len(idx)) < 0.25]
    df.loc[one_off, "beneficiary"] = [f"X{i:05d}" for i in range(len(one_off))]
    for col, drop in [("active_call", 0.3), ("screen_share", 0.3), ("new_app_installed", 0.4)]:
        has = idx[df.loc[idx, col] == 1]
        drop_rows = has[rng.random(len(has)) < drop]
        df.loc[drop_rows, col] = 0
    return df
if __name__ == "__main__":
    customers = make_customers(n_customers)
    df = make_genuine_transactions(customers, n_transactions)
    df = add_genuine_noise(df)
    df = add_genuine_overlap(df)
    df = inject_fraud(df)
    df = weaken_fraud(df)
    df["ben_is_merchant"] = df["beneficiary"].str.startswith("MER").astype(int)
    df = df.sort_values("timestamp").reset_index(drop = True)
    df.to_csv("data/transactions.csv", index = False)
    print("missing values: ", df.isna().sum().sum())
    print("AEPS rows: ", (df["channel"] == "AEPS").sum())
    print("G6 hours: ", sorted(df.loc[df["fraud_group"] == "G6", "timestamp"].dt.hour.unique()))
    print("G2 mules per victim: ", df[df["fraud_group"] == "G2"].groupby("customer_id")["beneficiary"].nunique().max())