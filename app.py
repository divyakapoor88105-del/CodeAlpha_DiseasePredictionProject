import streamlit as st
import pandas as pd
import os

# Page config
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="centered"
)

# Title section
st.markdown("<h1 style='text-align: center;'>🩺 Disease Prediction System</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 16px;'>"
    "A Machine Learning based system to compare classification models on medical datasets"
    "</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# Dataset selection
st.subheader("🗂 Dataset Selection")
dataset = st.selectbox(
    "Choose a medical dataset",
    ["Heart Disease", "Diabetes", "Breast Cancer"]
)

# Model selection
st.subheader("🤖 Model Selection")
model = st.selectbox(
    "Choose a machine learning model",
    ["Logistic_Regression", "SVM", "Random_Forest", "XGBoost"]
)

st.markdown("---")

# Show results
if st.button("🔍 View Model Evaluation"):
    dataset_key = dataset.replace(" ", "_")
    report_path = f"../results/{dataset_key}_{model}_report.txt"

    if os.path.exists(report_path):
        st.success("Model evaluation loaded successfully!")

        st.subheader("📊 Classification Report")
        with open(report_path, "r") as file:
            report = file.read()
        st.text(report)
    else:
        st.error("Evaluation report not found. Please run model training first.")

st.markdown("---")

# Comparison table
st.subheader("📈 Model Accuracy Comparison")
st.write(
    "The table below shows accuracy scores for all models across different datasets."
)

comparison_file = "../results/model_comparison.csv"
if os.path.exists(comparison_file):
    df = pd.read_csv(comparison_file, index_col=0)
    st.dataframe(df, use_container_width=True)
else:
    st.warning("Comparison file not available.")

st.markdown("---")

# Footer
st.markdown(
    "<p style='text-align: center; font-size: 14px;'>"
    "Developed using Python, Scikit-Learn, XGBoost & Streamlit"
    "</p>",
    unsafe_allow_html=True
)