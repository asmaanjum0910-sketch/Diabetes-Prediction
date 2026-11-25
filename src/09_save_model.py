
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load dataset
df = pd.read_csv(BASE_DIR / "data" / "diabetes.csv")

# Replace invalid zero values
zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

df[zero_columns] = df[zero_columns].replace(0, float("nan"))

# Separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Original best Decision Tree pipeline
best_model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            DecisionTreeClassifier(
                random_state=42,
                max_depth=4,
                min_samples_split=10,
                min_samples_leaf=5
            )
        )
    ]
)

# Train model
best_model.fit(X_train, y_train)

# Create models folder
models_dir = BASE_DIR / "models"
models_dir.mkdir(exist_ok=True)

# Save model
model_path = models_dir / "best_diabetes_model.pkl"

joblib.dump(best_model, model_path)

print("Model saved successfully!")
print("Saved location:", model_path)
