from pathlib import Path
import pandas as pd

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load dataset
df = pd.read_csv(BASE_DIR / "dataset" / "Student_performance_data.csv")

print("Original Shape:", df.shape)

# Remove StudentID
df = df.drop("StudentID", axis=1)

print("After Removing StudentID:", df.shape)

# Separate Features and Target
X = df.drop("GradeClass", axis=1)
y = df["GradeClass"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nFeature Columns:")
print(X.columns)