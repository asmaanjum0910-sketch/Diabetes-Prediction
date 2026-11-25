
import joblib
import pandas as pd

from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load saved model
model_path = BASE_DIR / "models" / "best_diabetes_model.pkl"

model = joblib.load(model_path)

print("Saved model loaded successfully!")

# Create sample patient records
new_patients = pd.DataFrame([
    {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 80,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.35,
        "Age": 30
    },
    {
        "Pregnancies": 6,
        "Glucose": 170,
        "BloodPressure": 85,
        "SkinThickness": 32,
        "Insulin": 150,
        "BMI": 35.0,
        "DiabetesPedigreeFunction": 0.75,
        "Age": 45
    }
])

# Predict
predictions = model.predict(new_patients)

# Get prediction probabilities
probabilities = model.predict_proba(new_patients)

# Display results
for i in range(len(new_patients)):

    print(f"\nPatient {i + 1}")

    print("Predicted class:", predictions[i])

    print(
        "Probability of class 0:",
        round(probabilities[i][0] * 100, 2), "%"
    )

    print(
        "Probability of class 1:",
        round(probabilities[i][1] * 100, 2), "%"
    )

    if predictions[i] == 1:
        print("Model output: Diabetes-positive class")
    else:
        print("Model output: Diabetes-negative class")
