import time
import pandas as pd
import streamlit as st
from src.batch import score_batch, required
from src.ui import style, inr
style()
st.title("Batch scoring")
st.caption("Learning fraud patterns from transactions behaviour")
@st.cache_data
def run(df):
    return score_batch(df)
with open("samples/sample_batch.csv", "rb") as f:
    st.download_button("Download a sample CSV", f, "sample_batch.csv", "text/csv")
    st.caption(f"Required columns: {', '.join(required)}. Other features default to 0. An optional is_fraud column turns on the accuracy check.")
use_sample = st.toggle("Use the sample file instead of uploading")
file = None if use_sample else st.file_uploader("Transactions CSV", type = "csv")
if use_sample:
    df = pd.read_csv("samples/sample_batch.csv")
elif file is not None:
    df = pd.read_csv(file)
else:
    st.stop()
if len(df) > 50000:
    st.warning("Only the first 50,000 rows are scored.")
    df = df.head(50000)
try:
    out = run(df)
except ValueError as e:
    st.error(str(e))
    st.stop()
counts = out["decision"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Payments scored", f"{len(out):,}")
c2.metric("Approved", f"{counts.get('Approve', 0):,}")
c3.metric("Step-up OTP", f"{counts.get('Step_up_OTP', 0):,}")
c4.metric("Blocked", f"{counts.get('Block', 0):,}", f"{inr(out.loc[out['decision'] == 'Block', 'amount'].sum())} held", delta_color = "off", delta_arrow = "off")
if "is_fraud" in out.columns:
    fraud = out["is_fraud"] == 1
    stopped = fraud & (out["decision"] != "Approve")
    st.info(f"Known fraud in this file: {fraud.sum()}, Stopped (blocked or OTP): {stopped.sum()} ({stopped.sum() / max(fraud.sum(), 1):.0%}). Genuine payments blocked: {((~fraud) & (out['decision'] == 'Block')).sum()}.")
st.subheader("Manual review queue")
queue = out[out["decision"] == "Block"].sort_values("fraud_probability", ascending = False)
st.dataframe(queue[["channel", "amount", "hour", "fraud_probability", "top_reason"]], hide_index = True, width = "stretch", column_config = {"fraud_probability" : st.column_config.ProgressColumn("Fraud probability", min_value = 0, max_value = 1, format = "%.2f"), "amount" : st.column_config.NumberColumn("Amount (₹)", format = "%.0f"), "top_reason" : st.column_config.TextColumn("Main reason", width = "large")})
st.download_button("Download all scored payments", out.to_csv(index = False).encode(), "fraudshield_scored.csv", "text/csv")
st.subheader("Replay as a live stream")
if st.button("Start replay"):
    board = st.empty()
    for i in range(1, 61):
        seen = out.head(i)
        with board.container():
            m1, m2, m3 = st.columns(3)
            m1.metric("Payments", i)
            m2.metric("Blocked", int((seen["decision"] == "Block").sum()))
            m3.metric("Step-up OTP", int((seen["decision"] == "Step_up_OTP").sum()))
            st.dataframe(seen.tail(6).iloc[::-1][["channel", "amount", "fraud_probability", "decision"]], hide_index = True, width = "stretch")
        time.sleep(0.25)