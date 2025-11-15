from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# --- Load data ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "subjects_processed.csv"

df = pd.read_csv(DATA_PATH)

y = df['group'].map({'CONTROL': 0, 'STROKE': 1})
df['ethnicity'] = df['ethnicity'].replace({'NON H/L': 'NON-H/L'})
df = df.drop(columns=["subject_number", "group"])

# Binary encoding
binary_cols = ["gender", "dm_patient_medical_history",
               "24hour_sbpdipper", "24hour_dbpdipper", "24hour_mbpdipper"]

multiclass_cols = ["ethnicity", "race"]
continuous_cols = [col for col in df.columns if col not in binary_cols + multiclass_cols]

for bcol in binary_cols:
    df[bcol] = df[bcol].map({"YES": 1, "NO": 0, "F": 0, "M": 1})

df = pd.get_dummies(df, columns=multiclass_cols, drop_first=True)

X = df.copy()

# --- Scale continuous features ---
scaler = StandardScaler()
X[continuous_cols] = scaler.fit_transform(X[continuous_cols])

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- PCA for dimensionality reduction ---
print("\n--- PCA Analysis ---")
pca = PCA(n_components=0.95)  # retain 95% of explained variance
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"Original feature count: {X_train.shape[1]}")
print(f"Reduced to {pca.n_components_} principal components explaining 95% variance")

# --- Feature importance proxy: contribution to first principal component ---
component_weights = np.abs(pca.components_[0])
top_features_idx = np.argsort(component_weights)[::-1][:10]
print("\nTop 10 contributing features to PC1:")
for i in top_features_idx:
    print(f"{X.columns[i]}  (weight = {component_weights[i]:.4f})")

# --- Train Linear SVM on PCA space ---
svm_linear = SVC(kernel="linear", class_weight="balanced", random_state=42)
svm_linear.fit(X_train_pca, y_train)

y_pred_linear = svm_linear.predict(X_test_pca)
print("\nLinear kernel SVM (on PCA) results:")
print(classification_report(y_test, y_pred_linear))

# --- Train RBF SVM with Grid Search on PCA space ---
svm_rbf = SVC(class_weight="balanced", random_state=42)

param_grid = {
    'gamma': [1, 0.1, 0.01],
    'C': [0.1, 1, 10],
    'kernel': ['rbf']
}

grid = GridSearchCV(svm_rbf, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train_pca, y_train)

print("\nBest params (RBF):", grid.best_params_)
print("Best CV F1:", grid.best_score_)

y_pred_rbf = grid.best_estimator_.predict(X_test_pca)
print("\nRBF kernel SVM (on PCA) results:")
print(classification_report(y_test, y_pred_rbf))
