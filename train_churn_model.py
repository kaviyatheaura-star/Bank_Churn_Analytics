"""
Train the churn prediction model for the European Bank dataset.

Usage:
    python train_churn_model.py European_Bank.csv

Outputs:
    churn_model.joblib     gradient-boosting model + preprocessing (for the app)
    churn_report.txt       metrics for both models and the logistic-regression odds ratios
"""

import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Exited"
DROP = ["CustomerId", "Surname", "Year"]
CATEGORICAL = ["Geography", "Gender"]
NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
           "HasCrCard", "IsActiveMember", "EstimatedSalary"]
RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
    ])


def evaluate(name, model, X_test, y_test, lines):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    lines.append(f"=== {name} ===")
    lines.append(f"ROC AUC:           {roc_auc_score(y_test, proba):.4f}")
    lines.append(f"Average precision: {average_precision_score(y_test, proba):.4f}")
    lines.append(f"Confusion matrix (rows=actual 0/1, cols=predicted 0/1):\n{confusion_matrix(y_test, pred)}")
    lines.append(classification_report(y_test, pred, target_names=["Retained", "Churned"], zero_division=0))
    return roc_auc_score(y_test, proba)


def main(csv_path: str) -> None:
    df = pd.read_csv(csv_path).drop(columns=DROP, errors="ignore")
    df = df.drop_duplicates().dropna()
    df[TARGET] = df[TARGET].astype(int)
    print(f"Loaded {len(df):,} rows. Churn rate: {df[TARGET].mean() * 100:.2f}%")

    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    lines = [f"Churn model report. Rows: {len(df):,}. Churn rate: {df[TARGET].mean() * 100:.2f}%.",
             f"Train: {len(X_train):,}  Test: {len(X_test):,} (stratified 80/20, seed {RANDOM_STATE}).", ""]

    # Gradient boosting model (main model)
    gb = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=4,
            class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    gb.fit(X_train, y_train)
    gb_auc = evaluate("Gradient boosting (main model)", gb, X_test, y_test, lines)

    # Logistic regression baseline (interpretable)
    lr = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    lr.fit(X_train, y_train)
    lr_auc = evaluate("Logistic regression (baseline)", lr, X_test, y_test, lines)

    # Odds ratios from the logistic regression (multivariate effect of each factor)
    names = lr.named_steps["prep"].get_feature_names_out()
    coefs = lr.named_steps["clf"].coef_[0]
    odds = pd.DataFrame({"feature": names, "odds_ratio": np.exp(coefs)})
    odds["feature"] = odds["feature"].str.replace(r"^(num|cat)__", "", regex=True)
    odds = odds.sort_values("odds_ratio", ascending=False)
    lines.append("=== Logistic regression odds ratios (per 1 SD for numeric, vs baseline for categorical) ===")
    lines.append(odds.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    lines.append("")
    lines.append(f"Summary: gradient boosting AUC {gb_auc:.3f} vs logistic regression AUC {lr_auc:.3f}.")
    lines.append("Odds ratio > 1 increases churn odds; < 1 decreases them.")

    joblib.dump({"model": gb, "features": NUMERIC + CATEGORICAL, "target": TARGET,
                 "threshold": 0.5}, "churn_model.joblib")
    with open("churn_report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print("\n".join(lines))
    print("\nSaved: churn_model.joblib and churn_report.txt")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python train_churn_model.py <path_to_csv>")
    main(sys.argv[1])
