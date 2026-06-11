# 🩺 Disease Prediction Using Machine Learning

## 📌 Project Overview
This project is a machine learning-based system that predicts diseases such as **Heart Disease, Diabetes, and Breast Cancer** using patient medical data like age, glucose level, cholesterol, and diagnostic features.

The system compares multiple ML models to analyze performance differences and improve prediction accuracy.

---

## ⚙️ Technologies Used
- Python 🐍
- Pandas & NumPy
- Scikit-learn
- XGBoost
- Streamlit (for UI)

---

## 🤖 Machine Learning Models
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

---

## 📊 How the Model Works
1. Medical datasets are loaded (CSV format)
2. Data is cleaned and preprocessed
3. Categorical labels (like B/M or Yes/No) are encoded into numeric format
4. Data is split into training and testing sets using stratified sampling
5. Multiple ML models are trained on the same dataset
6. Performance is evaluated using:
   - Accuracy
   - Precision
   - Recall
   - F1-score

---

## 📈 Why Different Metrics Occur

Different models show different performance because:
- Logistic Regression assumes linear relationships
- SVM finds optimal hyperplane boundaries
- Random Forest reduces variance using multiple decision trees
- XGBoost uses boosting to correct previous errors

Also, datasets were intentionally structured with:
- Non-linear relationships
- Noise in medical features
- Feature interactions

This ensures realistic performance variation similar to real-world medical data.

---

## 📂 Datasets Used
- Heart Disease Dataset
- Diabetes Dataset
- Breast Cancer Dataset

(Each dataset contains patient medical attributes and a target diagnosis column.)

---

## 🚀 How to Run the Project

```bash
pip install -r requirements.txt
cd src
python disease_prediction.py

To run UI:
streamlit run app.py
