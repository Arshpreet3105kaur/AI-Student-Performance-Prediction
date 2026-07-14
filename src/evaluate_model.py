from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt

# Project path
BASE_DIR = Path(__file__).resolve().parent.parent

# Load dataset
df = pd.read_csv(BASE_DIR / "dataset" / "Student_performance.csv")

# Remove StudentID
df = df.drop("StudentID", axis=1)

# Features and target
X = df.drop("GradeClass", axis=1)
y = df["GradeClass"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Load trained model
model = joblib.load(BASE_DIR / "models" / "student_model.pkl")

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("\n========== ACCURACY ==========")
print(accuracy_score(y_test, y_pred))

# Classification Report
print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\n========== CONFUSION MATRIX ==========")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature Importance
importance = model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_df)

# Plot
plt.figure(figsize=(8,6))

plt.barh(
    feature_df["Feature"],
    feature_df["Importance"]
)

plt.xlabel("Importance")
plt.title("Feature Importance")

plt.gca().invert_yaxis()

# Create screenshots folder
(BASE_DIR / "screenshots").mkdir(exist_ok=True)

plt.savefig(BASE_DIR / "screenshots" / "feature_importance.png")

plt.show()