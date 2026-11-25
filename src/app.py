
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load trained model
MODEL_PATH = BASE_DIR / "models" / "best_diabetes_model.pkl"
model = joblib.load(MODEL_PATH)

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# Application heading
st.title("🩺 Diabetes Prediction System")

st.write(
    "Enter the required information to see the "
    "machine learning model's prediction."
)

st.warning(
    "Educational demonstration only. "
    "This application cannot diagnose diabetes."
)

# Input form
with st.form("prediction_form"):

    st.subheader("Patient Information")

    pregnancies = st.number_input(
        "Number of Pregnancies",
        min_value=0,
        max_value=20,
        value=2
    )

    glucose = st.number_input(
        "Glucose Level",
        min_value=1.0,
        max_value=300.0,
        value=120.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=1.0,
        max_value=200.0,
        value=70.0
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=1.0,
        max_value=100.0,
        value=25.0
    )

    insulin = st.number_input(
        "Insulin Level",
        min_value=1.0,
        max_value=1000.0,
        value=80.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=1.0,
        max_value=80.0,
        value=28.5
    )

    pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.35
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

    submitted = st.form_submit_button(
        "Predict"
    )

# Make prediction
if submitted:

    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age
    }])

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Model output: Diabetes-positive class")
    else:
        st.success("Model output: Diabetes-negative class")

    st.write(
        f"Class 0 probability: {probabilities[0] * 100:.2f}%"
    )

    st.write(
        f"Class 1 probability: {probabilities[1] * 100:.2f}%"
    )
