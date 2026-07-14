from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load dataset
df = pd.read_csv(BASE_DIR / "dataset" / "Student_performance_data.csv")

# Remove StudentID
df = df.drop("StudentID", axis=1)

# Features and Target
X = df.drop("GradeClass", axis=1)
y = df["GradeClass"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Create models folder if it doesn't exist
models_dir = BASE_DIR / "models"
models_dir.mkdir(exist_ok=True)

# Save model
joblib.dump(model, models_dir / "student_model.pkl")

print("\nModel saved successfully!")