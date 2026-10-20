import time
import streamlit as st
from src.api import Transaction, to_row, decide, model, iso, iso_cut, explainer, block_threshold, cost_threshold
from src.explain import explain_one
from src.scenarios import scenarios, reason_text
colours = {"Block" : "#E5484D", "Step_up_OTP" : "#F5A524", "Approve" : "#30A46C"}
labels = {"Block" : "BLOCK", "Step_up_OTP" : "STEP-UP OTP", "Approve" : "APPROVE"}
st.markdown("""<style>
@keyframes rise {from {opacity : 0 ; transform : translateY(8px);} to {opacity : 1 ; transform : None}}
.verdict {border-radius : 14px ; padding : 18px 22px ; animation : rise 0.5s ease-out;}
.phone {width : 230px ; margin : auto ; border : 1px solid #3A3F4B ; border-radius : 28px ; padding : 22px 16px ; background : #111520 ; animation : rise 0.6s ease-out;}
</style>""", unsafe_allow_html = True)
st.title("FraudShield")
st.caption("AI-based online fraudulent payment detection system")
choice = st.pills("Scenario", list(scenarios), default = list(scenarios)[0])
sc = scenarios[choice]
st.write(sc["story"])
tx = dict(sc["tx"])
with st.expander("What if? Adjust the payment"):
    c1, c2, c3 = st.columns(3)
    tx["amount_ratio"] = c1.slider("x usual amount", 0.2, 10.0, float(tx["amount_ratio"]), 0.1)
    tx["ben_fan_in"] = c1.slider("People who paid this account", 0, 150, int(tx.get("ben_fan_in", 0)))
    tx["active_call"] = int(c2.toggle("Phone call active", bool(tx.get("active_call", 0))))
    tx["screen_share"] = int(c2.toggle("Screen share done", bool(tx.get("screen_share", 0))))
    tx["is_new_beneficiary"] = int(c2.toggle("New account holder", bool(tx.get("is_new_beneficiary", 0))))
    tx["new_app_installed"] = int(c3.toggle("New app installed", bool(tx.get("new_app_installed", 0))))
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
    st.caption(f"Step-up OTP from {cost_threshold}, Block from {block_threshold:.2f}")
    st.subheader("Why")
    if d == "Approve" or not reasons:
        st.markdown("Nothing unusual : a familiar payee, a normal amount and a normal time.")
    else:
        for f in reasons:
            st.markdown(f"- {reason_text(f, tx)}")
with right:
    if d != "Approve" or choice == list(scenarios)[0]:
        msg = sc["phone"]
    else:
        msg = "Payment approved."
    st.markdown(f"<div class = 'phone'>"
                f"<div style = 'text-align : center ; opacity : 0.6 ; font-size : 12px'>Customer's phone</div>"
                f"<div style = 'margin-top : 14px ; padding : 12px ; border-radius : 12px ; background : {colours[d]}22 ; border-left : 3px solid {colours[d]}'>{msg}</div>"
                f"</div>", unsafe_allow_html = True)