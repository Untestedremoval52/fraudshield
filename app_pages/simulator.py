import time
import streamlit as st
from src.api import Transaction, to_row, decide, model, iso, iso_cut, explainer, block_threshold, cost_threshold
from src.explain import explain_one
from src.scenarios import reason_text
from src.situations import situations
colours = {"Block" : "#E5484D", "Step_up_OTP" : "#F5A524", "Approve" : "#30A46C"}
labels = {"Block" : "BLOCK", "Step_up_OTP" : "STEP-UP OTP", "Approve" : "APPROVE"}
st.markdown("""<style>
@keyframes rise {from {opacity : 0 ; transform : translateY(8px);} to {opacity : 1 ; transform : none}}
.verdict {border-radius : 14px ; padding : 18px 22px ; animation : rise 0.5s ease-out;}
.phone {width : 230px ; margin : auto ; border : 1px solid #3A3F4B ; border-radius : 28px ; padding : 22px 16px ; background : #111520 ; animation : rise 0.6s ease-out;}
[data-testid = "InputInstructions"] {display : none;}
</style>""", unsafe_allow_html = True)
st.title("FraudShield")
st.caption("AI-based online fraudulent payment detection system")
with st.form("payment"):
    a, b = st.columns(2, gap = "large")
    with a:
        st.markdown("**1. Your payment**")
        amount = st.number_input("Amount (₹)", min_value = 1, max_value = 1000000, value = None, placeholder = "5000")
        usual = st.number_input("Your usual payment amount (₹)", min_value = 1, max_value = 1000000, value = None, placeholder = "2000")
        channel = st.selectbox("Paid with", ["UPI", "Card_online", "Net_banking", "ATM"], index = None, placeholder = "Choose a method")
        category = st.selectbox("Paying for", ["grocery", "food", "travel", "electronics", "utilities", "fashion", "entertainment", "transfer"], index = None, placeholder = "Choose a category")
        hour = st.slider("Hour of the day", 0, 23, 14)
    with b:
        st.markdown("**2. What is happening around it**")
        situation = st.radio("Situation", list(situations), index = None, captions = [s["story"] for s in situations.values()], label_visibility = "collapsed")
    submitted = st.form_submit_button("Check payment", type = "primary")
if submitted:
    if None in (amount, usual, channel, category, situation):
        st.error("Fill in all the steps 1 and 2 first.")
        st.session_state.pop("tx", None)
    else:
        st.session_state["tx"] = {"channel" : channel, "merchant_category" : category, "amount" : amount, "amount_ratio" : round(amount / usual, 2), "hour" : hour, **situations[situation]["signals"]}
        st.session_state["situation"] = situation
if "tx" not in st.session_state:
    st.stop()
tx, sc = st.session_state["tx"], situations[st.session_state["situation"]]
start = time.perf_counter()
row = to_row(Transaction(**tx))
p = float(model.predict_proba(row)[0, 1])
anomalous = float(-iso.decision_function(row)[0]) >= iso_cut
d = decide(p, anomalous)
reasons = [f for f, v in explain_one(model, explainer, row) if v >= 0.1]
ms = (time.perf_counter() - start) * 1000
left, right = st.columns([3, 2], gap = "large")
with left:
    info = f"Fraud probability {p:.2f} and decided in {ms:.0f} ms"
    if d == "Block":
        info += " · Sent for manual reviewing"
    if anomalous:
        info += " · Unusual pattern flagged"
    st.markdown(f"<div class = 'verdict' style = 'background : {colours[d]}22 ; border : 1px solid {colours[d]}'>"
                f"<div style = 'font-size : 30px ; font-weight : 600 ; color : {colours[d]}'>{labels[d]}</div>"
                f"<div style = 'opacity : 0.8'>{info}</div>" f"</div>", unsafe_allow_html = True)
    st.progress(p)
    st.subheader("Why")
    if d == "Approve" or not reasons:
        st.markdown("Nothing that looks like a fraud stands out.")
    else:
        for f in reasons:
            st.markdown(f"- {reason_text(f, tx)}")
with right:
    amt = f"₹{tx['amount']:,.0f}"
    if d == "Block":
        msg = f"Payment of {amt} blocked. {sc['hint']}"
    elif d == "Step_up_OTP":
        msg = f"Did you start this {amt} payment? {sc['hint']} Confirm with your OTP."
    else:
        msg = f"{amt} paid."
    st.markdown(f"<div class = 'phone'>"
                f"<div style = 'text-align : center ; opacity : 0.6 ; font-size : 12px'>Customer's phone</div>"
                f"<div style = 'margin-top : 14px ; padding : 12px ; border-radius : 12px ; background : {colours[d]}22 ; border-left : 3px solid {colours[d]}'>{msg}</div>"
                f"</div>", unsafe_allow_html = True)