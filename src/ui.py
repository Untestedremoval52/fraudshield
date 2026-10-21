import streamlit as st
colours = {"Block" : "#E5484D", "Step_up_OTP" : "#F5A524", "Approve" : "#30A46C"}
labels = {"Block" : "BLOCK", "Step_up_OTP" : "STEP-UP OTP", "Approve" : "APPROVE"}
css = """<style>
@keyframes rise {from {opacity : 0 ; transform : translateY(8px);} to {opacity : 1 ; transform : none}}
.verdict {border-radius : 14px ; padding : 18px 22px ; animation : rise 0.5s ease-out ;}
.phone {width : 230px ; margin : auto ; border : 1px solid #3A3F4B ; border-radius : 28px ; padding : 22px 16px ; background : #111520 ; animation : rise 0.6s ease-out ;}
[data-testid = "InputInstructions"] {display : none ;}
[data-testid = "stMetric"] {background : #1A1F2B ; border-radius : 12px ; padding : 14px 16px ;}
</style>"""
def style():
    st.markdown(css, unsafe_allow_html = True)
def inr(x):
    s = str(int(round(abs(x))))
    if len(s) > 3:
        head, tail = s[ : -3], s[-3 : ]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2 : ])
            head = head[ : -2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups) + "," + tail
    return ("-" if x < 0 else "") + "₹" + s
pretty = {"Approve" : "Approved", "Step_up_OTP" : "Step-up OTP", "Block" : "Blocked"}
def table_columns():
    return {"channel" : st.column_config.TextColumn("Channel"), "amount" : st.column_config.NumberColumn("Amount",format = "₹%d"), "hour" : st.column_config.NumberColumn("Hour", format = "%d:00"), "fraud_probability" : st.column_config.ProgressColumn("Fraud probability", min_value = 0, max_value = 1, format = "%.2f"), "top_reason" : st.column_config.TextColumn("Main reason", width = "large"), "decision" : st.column_config.TextColumn("Decision")}