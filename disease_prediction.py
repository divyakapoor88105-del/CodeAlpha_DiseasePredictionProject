import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# -------------------------------
# LOAD DATA WITH AUTO TARGET FIX
# -------------------------------
def load_data(csv_path, target_candidates):
    data = pd.read_csv(csv_path)

    # Find target column automatically
    target_column = None
    for col in data.columns:
        if col.lower() in target_candidates:
            target_column = col
            break

    if target_column is None:
        raise ValueError(
            f"❌ Target column not found in {csv_path}\n"
            f"Available columns: {list(data.columns)}"
        )

    X = data.drop(columns=[target_column])
    y = data[target_column]

    return X, y, target_column


# -------------------------------
# TRAIN & EVALUATE
# -------------------------------
from sklearn.preprocessing import LabelEncoder


def train_and_evaluate(X, y, dataset_name):
    # Encode target if it is categorical (B/M, Yes/No etc.)
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"🔁 Encoded target classes: {list(le.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=42
        )
    }

    for name, model in models.items():
        print(f"\n{dataset_name} - {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))


# -------------------------------
# MAIN
# -------------------------------
def main():
    datasets = {
        "Heart Disease": "../data/heart.csv",
        "Diabetes": "../data/diabetes.csv",
        "Breast Cancer": "../data/breast_cancer.csv"
    }

    # possible names of target column (LOWERCASE)
    target_candidates = {
        "target", "outcome", "class", "label",
        "result", "diagnosis", "diabetes"
    }

    for name, path in datasets.items():
        print(f"\n===== Processing {name} Dataset =====")

        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            continue

        X, y, target_col = load_data(path, target_candidates)
        print(f"✅ Detected target column: {target_col}")

        train_and_evaluate(X, y, name)


# -------------------------------
if __name__ == "__main__":
    main()