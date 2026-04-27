import streamlit as st
import requests

st.title("💳 Credit Card Compliance Checker")

interest = st.number_input("Interest Rate (%)", 0.0, 100.0)
late_fee = st.number_input("Late Payment Fee (₹)", 0.0)
annual_fee = st.number_input("Annual Fee (₹)", 0.0)
billing_cycle = st.number_input("Billing Cycle (days)", 0)
min_payment = st.number_input("Minimum Payment (%)", 0.0, 100.0)
disclosure = st.selectbox("Disclosure Provided", ["Yes", "No"])

if st.button("Check Compliance"):

    payload = {
        "interest_rate": interest,
        "late_fee": late_fee,
        "annual_fee": annual_fee,
        "billing_cycle": billing_cycle,
        "min_payment": min_payment,
        "disclosure": 1 if disclosure == "Yes" else 0
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=payload)

    result = response.json()

    st.subheader(f"Prediction: {result['prediction']}")
    st.write(f"Confidence: {round(result['confidence']*100, 2)}%")