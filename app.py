import streamlit as st
st.set_page_config(page_title = "FraudShield", page_icon = ":material/shield:", layout = "wide")
pages = [st.Page("app_pages/simulator.py", title = "Payment simulator", icon = ":material/payments:"), st.Page("app_pages/batch.py", title = "Batch scoring", icon = ":material/table:"), st.Page("app_pages/performance.py", title = "Model performance", icon = ":material/monitoring:")]
st.navigation(pages).run()