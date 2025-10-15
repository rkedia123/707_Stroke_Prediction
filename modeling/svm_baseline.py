from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Always start path relative to this script file
BASE_DIR = Path(__file__).resolve().parent.parent   # go up to repo root
DATA_PATH = BASE_DIR / "data" / "processed" / "subjects_processed.csv"

df = pd.read_csv(DATA_PATH)

y = df['group'].map({'CONTROL': 0, 'STROKE': 1})

df['ethnicity'] = df['ethnicity'].replace({'NON H/L': 'NON-H/L'})

# Binary encodings
binary_cols = ["gender", "dm_patient_medical_history",
               "24hour_sbpdipper", "24hour_dbpdipper", "24hour_mbpdipper"]

for bcol in binary_cols:
    df[bcol] = df[bcol].map({"YES": 1, "NO": 0, "F": 0, "M": 1})

# One-hot encoding for multiclass
df = pd.get_dummies(df, columns=["ethnicity", "race"], drop_first=True)

X = df.drop(columns=["subject_number", "group"])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train a linear SVM
svm = SVC(kernel="linear", class_weight="balanced", random_state=42)
svm.fit(X_train, y_train)

# Evaluate
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))




