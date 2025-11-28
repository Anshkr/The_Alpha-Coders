import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="E-commerce Fraud Detector", layout="wide")
st.title("🛡️ E-commerce Fraud Detection App")
st.write(
    "Upload your transaction data to get instant fraud predictions, "
    "with visual insights and downloadable results."
)

# ==============================
# LOAD MODEL + PIPELINE PARTS
# ==============================
model = joblib.load("fraud_model.pkl")
num_imputer = joblib.load("num_imputer.pkl")
cat_imputer = joblib.load("cat_imputer.pkl")
encoder = joblib.load("target_encoder.pkl")
scaler = joblib.load("scaler.pkl")
ALL_FEATURES = joblib.load("model_columns.pkl")

# ==============================
# FILE UPLOAD
# ==============================
st.markdown("## 📥 Upload Transactions")
uploaded_file = st.file_uploader("Upload CSV file with transaction data", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Drop unnecessary columns
    DROP_COLS = ["fraud_prob_hidden", "user_id"]
    for col in DROP_COLS:
        if col in data.columns:
            data = data.drop(columns=[col])

    # Preview top 5 rows & top 5 columns
    preview_cols = data.columns[:5]
    st.markdown("### 📄 Uploaded Data Preview (Top 5 Rows & Columns)")
    st.dataframe(data[preview_cols].head(), use_container_width=True)

    # Identify column types
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()

    # ==============================
    # DATA PREPROCESSING
    # ==============================
    if numerical_cols:
        data[numerical_cols] = num_imputer.transform(data[numerical_cols])
    if categorical_cols:
        data[categorical_cols] = cat_imputer.transform(data[categorical_cols])
    data = encoder.transform(data)
    data[ALL_FEATURES] = scaler.transform(data[ALL_FEATURES])

    # ==============================
    # PREDICTION
    # ==============================
    pred = model.predict(data)
    data["fraud_prediction"] = pred
    fraud_count = int(sum(pred))
    total = len(pred)

    # ==============================
    # CREATE TABS
    # ==============================
    tab1, tab2 = st.tabs(["📄 Predictions Table", "📊 Visualizations"])

    # ------------------------------
    # TAB 1: Predictions Table + Download
    # ------------------------------
    with tab1:
        st.markdown("### Predictions Table")
        st.dataframe(data, use_container_width=True)

        st.download_button(
            label="💾 Download Predictions CSV",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="fraud_results.csv",
            mime="text/csv"
        )

        st.markdown(f"### ⚠️ Fraudulent Transactions: {fraud_count} / {total}")
        st.progress(fraud_count / total)

    # ------------------------------
    # TAB 2: Visualizations
    # ------------------------------
    with tab2:
        st.markdown("### 📊 Fraud vs Legit Transactions")
        fraud_counts = pd.Series(pred).value_counts().rename({0: "Legit", 1: "Fraud"})
        fig_pie = px.pie(
            names=fraud_counts.index,
            values=fraud_counts.values,
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Optional: Interactive selection for categorical feature
        if categorical_cols:
            st.markdown("### 🚨 Fraud Distribution by Categorical Feature")
            selected_col = st.selectbox("Select feature to visualize", categorical_cols)
            fraud_by_col = data.groupby(selected_col)["fraud_prediction"].sum().sort_values(ascending=False)
            fig_bar = px.bar(
                x=fraud_by_col.index,
                y=fraud_by_col.values,
                labels={'x': selected_col, 'y': 'Number of Fraud Transactions'},
                color=fraud_by_col.values,
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.info("Please upload a CSV file to begin. Supported file type: `.csv`")
