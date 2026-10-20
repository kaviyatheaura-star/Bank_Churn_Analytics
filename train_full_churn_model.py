"""
Full training of the churn model on ALL rows.
- 5-fold stratified cross-validation for reliable metrics
- Final model refit on 100% of the data and saved
Usage: python train_full_churn_model.py European_Bank.csv
"""
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Exited"
DROP = ["CustomerId", "Surname", "Year"]
CAT = ["Geography", "Gender"]
NUM = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
       "HasCrCard", "IsActiveMember", "EstimatedSalary"]

def prep():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])

def gb():
    return Pipeline([("prep", prep()),
                     ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                            max_depth=4, class_weight="balanced",
                                                            random_state=42))])
def lr():
    return Pipeline([("prep", prep()),
                     ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])

def main(path):
    df = pd.read_csv(path).drop(columns=DROP, errors="ignore").drop_duplicates().dropna()
    df[TARGET] = df[TARGET].astype(int)
    X, y = df[NUM + CAT], df[TARGET]
    print(f"Rows: {len(df):,}  Churn rate: {y.mean()*100:.2f}%")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"roc_auc": "roc_auc", "avg_precision": "average_precision",
               "precision": "precision", "recall": "recall", "f1": "f1"}
    lines = [f"Full churn model training. Rows: {len(df):,}. Churn rate: {y.mean()*100:.2f}%.",
             "", "5-fold stratified cross-validation (mean ± std across folds):"]
    for name, mk in [("Gradient boosting (final model)", gb), ("Logistic regression (baseline)", lr)]:
        res = cross_validate(mk(), X, y, cv=cv, scoring=scoring)
        lines.append(f"\n{name}:")
        for k in scoring:
            v = res[f"test_{k}"]
            lines.append(f"  {k:<14} {v.mean():.4f} ± {v.std():.4f}")

    final = gb().fit(X, y)
    lines.append("\nFinal gradient-boosting model refit on 100% of rows.")
    lines.append("In-sample scores (for reference only; cross-validation above is the honest estimate):")
    p = final.predict_proba(X)[:, 1]
    lines.append(f"  ROC AUC {roc_auc_score(y, p):.4f}, average precision {average_precision_score(y, p):.4f}")

    joblib.dump({"model": final, "features": NUM + CAT, "target": TARGET, "threshold": 0.5,
                 "trained_rows": len(df)}, "churn_model_full.joblib")
    open("churn_full_report.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nSaved: churn_model_full.joblib, churn_full_report.txt")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python train_full_churn_model.py <csv>")
    main(sys.argv[1])
