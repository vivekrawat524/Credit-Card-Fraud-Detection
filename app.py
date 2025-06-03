import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('fraud_detection_model.pkl')

st.title("💳 Credit Card Fraud Detection")

st.markdown("Enter transaction details below:")

# Create input fields for V1 to V28 (you can reduce inputs if needed)
input_fields = ['scaled_time', 'scaled_amount'] + [f'V{i}' for i in range(1, 29)]
user_input = []

for field in input_fields:
    value = st.number_input(f"Enter {field}", value=0.0)
    user_input.append(value)

# Predict button
if st.button("Predict"):
    prediction = model.predict([user_input])
    if prediction[0] == 1:
        st.error("⚠️ Fraud Detected!")
    else:
        st.success("✅ Transaction is Legitimate.")
