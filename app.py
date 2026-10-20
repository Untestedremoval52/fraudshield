import streamlit as st
st.set_page_config(page_title = "FraudShield", page_icon = ":material/shield:", layout = "wide")
pages = [st.Page("app_pages/simulator.py", title = "Payment simulator", icon = ":material/payments:")]
st.navigation(pages).run()