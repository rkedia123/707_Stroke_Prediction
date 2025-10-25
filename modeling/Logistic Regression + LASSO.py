#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 13:38:32 2025

@author: wzliu
"""

# ==========================================
# Stroke vs Control Classification
# Logistic Regression (L2) + LASSO (L1)
# ==========================================
# Output:
#  - Summary metrics
#  - Optimal threshold (Youden's J)
#  - LASSO selected features
#  - Predicted probabilities
#  - ROC curves (PNG)
# ==========================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
import statsmodels.api as sm

# -------------------------------
# 0. I/O Settings
# -------------------------------
INPATHS = ["subjects_processed.csv", "/Users/wzliu/Desktop/subjects_processed.csv"]
for p in INPATHS:
    if os.path.exists(p):
        INPATH = p
        break
else:
    raise FileNotFoundError("CSV not found. Move it next to this script or Desktop.")

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# -------------------------------
# 1. Load Data
# -------------------------------
df = pd.read_csv(INPATH)
df["stroke"] = df["group"].str.contains("STROKE", case=False).astype(int)

# 将非数值类型转为 one-hot (drop_first 避免冗余编码)
cat_cols = [c for c in df.select_dtypes(include="object").columns if c not in ["subject_number", "group"]]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Features & Target
X = df.drop(columns=["subject_number", "group", "stroke"])
y = df["stroke"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# -------------------------------
# 2. Metrics Helper
# -------------------------------
def eval_at_threshold(y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()

    return {
        "Threshold": threshold,
        "AUC": roc_auc_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, pred),
        "Sensitivity": tp / max(1, tp + fn),
        "Specificity": tn / max(1, tn + fp),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn
    }

# -------------------------------
# 3. Logistic Regression (L2)
# -------------------------------
logit_l2 = make_pipeline(
    StandardScaler(with_mean=False),
    LogisticRegression(penalty="l2", max_iter=4000)
)
logit_l2.fit(X_train, y_train)
yprob_l2 = logit_l2.predict_proba(X_test)[:, 1]

# -------------------------------
# 4. LASSO Logistic (L1 + CV)
# -------------------------------
lasso = make_pipeline(
    StandardScaler(with_mean=False),
    LogisticRegressionCV(
        penalty="l1", solver="liblinear",
        cv=5, Cs=10, scoring="roc_auc", max_iter=4000
    )
)
lasso.fit(X_train, y_train)
yprob_l1 = lasso.predict_proba(X_test)[:, 1]

# -------------------------------
# 5. Metrics @ threshold = 0.5
# -------------------------------
results_05 = pd.DataFrame([
    {"Model": "Logistic (L2)", **eval_at_threshold(y_test, yprob_l2, 0.5)},
    {"Model": "LASSO (L1)", **eval_at_threshold(y_test, yprob_l1, 0.5)},
])
print("\n===== Performance @ Threshold = 0.5 =====")
print(results_05.round(3))
results_05.to_csv(f"{OUTDIR}/metrics_threshold_0.5.csv", index=False)

# -------------------------------
# 6. Optimal Threshold (Youden's J)
# -------------------------------
def optimal_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    idx = np.argmax(tpr + (1 - fpr) - 1)
    return thr[idx]

thr_l2 = optimal_threshold(y_test, yprob_l2)
thr_l1 = optimal_threshold(y_test, yprob_l1)

results_opt = pd.DataFrame([
    {"Model": "Logistic (L2)", **eval_at_threshold(y_test, yprob_l2, thr_l2)},
    {"Model": "LASSO (L1)", **eval_at_threshold(y_test, yprob_l1, thr_l1)},
])
print("\n===== Performance @ Optimal Threshold =====")
print(results_opt.round(3))
results_opt.to_csv(f"{OUTDIR}/metrics_optimal.csv", index=False)

# -------------------------------
# 7. LASSO Selected Features
# -------------------------------
coef = lasso.named_steps["logisticregressioncv"].coef_.ravel()
lasso_features = pd.DataFrame({
    "Feature": X.columns,
    "Coef": coef,
    "OR": np.exp(coef)
})
lasso_features = lasso_features[lasso_features["Coef"].abs() > 1e-9]
lasso_features = lasso_features.sort_values("Coef", ascending=False)

print("\n===== LASSO Selected Predictors =====")
print(lasso_features.round(4))
lasso_features.to_csv(f"{OUTDIR}/lasso_features.csv", index=False)

# -------------------------------
# 8. Save Predictions
# -------------------------------
pred_out = pd.DataFrame({
    "True": y_test.values,
    "Prob_L2": yprob_l2,
    "Prob_L1": yprob_l1
})
pred_out.to_csv(f"{OUTDIR}/predictions.csv", index=False)

# -------------------------------
# 9. ROC Curve Plot
# -------------------------------
fpr2, tpr2, _ = roc_curve(y_test, yprob_l2)
fpr1, tpr1, _ = roc_curve(y_test, yprob_l1)

plt.figure()
plt.plot(fpr2, tpr2, label=f"L2 (AUC={roc_auc_score(y_test,yprob_l2):.3f})")
plt.plot(fpr1, tpr1, label=f"L1 (AUC={roc_auc_score(y_test,yprob_l1):.3f})")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("ROC Comparison")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/roc_curves.png", dpi=200)
plt.close()

print(f"\n Done! Outputs saved to: {OUTDIR}/")
