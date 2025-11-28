import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

st.set_page_config(page_title="E-commerce Fraud Detector", layout="wide")

st.title("🛡️ E-commerce Fraud Detection App")
st.write("Upload transaction data and get fraud predictions instantly.")

# Load model
model = joblib.load("fraud_model.pkl")

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])


if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### 📄 Uploaded Data")
    st.dataframe(data.head())

    # Predict
    pred = model.predict(data)

    data["fraud_prediction"] = pred

    st.write("### 🔍 Predictions")
    st.dataframe(data)

    # Download
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name="fraud_results.csv",
        mime="text/csv",
    )

    # Fraud summary
    fraud_count = sum(pred)
    total = len(pred)

    st.write(f"### ⚠️ Fraudulent Transactions: {fraud_count} / {total}")
    st.progress(fraud_count / total)
else:
    st.info("Please upload a CSV file to begin.")
