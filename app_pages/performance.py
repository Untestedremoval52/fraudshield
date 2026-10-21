import ast
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score
from src.api import block_threshold, cost_threshold
from src.ui import style, inr
style()
st.title("Model performance")
st.caption("Every number here comes from payments the model never saw during training.")
pred = pd.read_csv("reports/test_predictions.csv")
y, p, amt = pred["is_fraud"].to_numpy(), pred["fraud_probability"].to_numpy(), pred["amount"].to_numpy()
st.subheader("Pick a threshold, see the trade-off")
st.session_state.setdefault("t", 0.01)
def jump(value):
    st.session_state["t"] = value
b1, b2, b3 = st.columns(3)
b1.button("Flag everything", on_click = jump, args = (0.01,))
b2.button("Cheapest setting (step-up OTP)", on_click = jump, args = (0.18,))
b3.button("Best F1 setting (block)", on_click = jump, args = (round(block_threshold, 2),))
c1, c2 = st.columns(2)
t = c1.slider("Flag a payment when the fraud probability is at least", 0.01, 0.99, step = 0.01, key = "t")
fa_cost = c2.select_slider("Cost of one false alarm (₹)", [50, 100, 200, 500, 1000], 200, help = "What one genuine payment wrongly flagged costs the bank: the OTP, a support call, an annoyed customer.")
flag = p >= t
caught = int((flag & (y == 1)).sum())
missed = int((~flag & (y == 1)).sum())
false_alarms = int((flag & (y == 0)).sum())
cost = amt[~flag & (y == 1)].sum() + fa_cost * false_alarms
m1, m2, m3, m4 = st.columns(4)
m1.metric("Fraud caught", f"{caught} of {caught + missed}")
m2.metric("Fraud missed", missed)
m3.metric("False alarms", f"{false_alarms:,}")
m4.metric("Total cost", inr(cost))
st.caption(f"PR-AUC {average_precision_score(y, p):.3f} · Precision {caught / max(caught + false_alarms, 1):.2f} · Recall {caught / max(caught + missed, 1):.2f} · Block threshold {block_threshold:.2f} · Cost threshold {cost_threshold}")
grid = np.linspace(0.01, 0.95, 95)
curve = pd.DataFrame({"threshold" : grid, "total cost (₹)" : [amt[(p < g) & (y == 1)].sum() + fa_cost * ((p >= g) & (y == 0)).sum() for g in grid]}).set_index("threshold")
st.line_chart(curve)
st.subheader("Fraud it has never seen")
unseen = pd.read_csv("reports/unseen_fraud.csv").set_index("detector")
st.bar_chart(unseen["unseen_G6_caught"], horizontal = True)
st.caption("Malware fraud was never shown to the model during training. Adding Isolation Forest raises how much of it is caught from 36% to 79%.")
st.subheader("Choosing the model")
cv = pd.read_csv("reports/cv_selection.csv")
cv["folds"] = cv["folds"].apply(ast.literal_eval)
st.dataframe(cv, hide_index = True, width = "stretch", column_config = {"folds" : st.column_config.LineChartColumn("PR-AUC per time fold", y_min = 0.5, y_max = 1.0)})
st.subheader("Why the split method matters")
split = pd.read_csv("reports/split_comparison.csv").pivot(index = "model", columns = "split", values = "pr_auc")
st.bar_chart(split, stack = False)
st.caption("A random split lets the model peek at the future data, which inflates its scores. The time split tests it the way a bank would use it: on payments that happen later.")
st.subheader("What drives the decision")
st.image("reports/figures/shap_summary.png")
st.caption("Accounts paid by a handful of different people push the score up (the mule pattern), accounts paid by hundreds, like shops, push it down.")
st.subheader("Check on the real-world-type data (PaySim)")
st.dataframe(pd.read_csv("reports/paysim_results.csv"), hide_index = True, width = "stretch")
st.subheader("Limitations")
st.warning("This is based on synthetic data, PaySim artefact and has a large-mule-network blind spot")