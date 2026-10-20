import warnings, joblib, pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("European_Bank.csv").drop(columns=["CustomerId","Surname","Year"], errors="ignore")
df = df.drop_duplicates().dropna()
NUM = ["CreditScore","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]
CAT = ["Geography","Gender"]
X, y = df[NUM+CAT], df["Exited"].astype(int)

pre = ColumnTransformer([("num", StandardScaler(), NUM),
                         ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])
pipe = Pipeline([("prep", pre), ("clf", HistGradientBoostingClassifier(
    class_weight="balanced", random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20))])
grid = {"clf__learning_rate":[0.02,0.05,0.1], "clf__max_depth":[2,3,4,6],
        "clf__max_leaf_nodes":[15,31], "clf__l2_regularization":[0.0,1.0], "clf__min_samples_leaf":[20,50]}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    search.fit(X, y)
print("Warnings during search:", len(w))
for item in set(str(x.message)[:150] for x in w): print(" -", item)
print("Best params:", search.best_params_)
print(f"Best CV ROC AUC: {search.best_score_:.4f}")
# CV metrics for the best config, with the same folds
from sklearn.model_selection import cross_validate
res = cross_validate(search.best_estimator_, X, y, cv=cv,
    scoring={"roc_auc":"roc_auc","avg_precision":"average_precision","precision":"precision","recall":"recall","f1":"f1"})
for k in ["roc_auc","avg_precision","precision","recall","f1"]:
    v=res[f"test_{k}"]; print(f"{k:<14} {v.mean():.4f} ± {v.std():.4f}")
joblib.dump({"model": search.best_estimator_, "features": NUM+CAT, "target":"Exited",
             "threshold":0.5, "best_params": search.best_params_}, "churn_model_tuned.joblib")
