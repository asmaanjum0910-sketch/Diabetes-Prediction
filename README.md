# Diabetes Prediction Using Machine Learning

## 1. Project Overview

This project uses Machine Learning classification algorithms to predict whether a person is likely to have diabetes based on selected medical and demographic measurements.

The project demonstrates an end-to-end Machine Learning workflow, including:
- Data understanding and exploratory data analysis (EDA)
- Data cleaning and preprocessing
- Model training and comparison
- Hyperparameter tuning
- Model evaluation
- Model saving and loading
- Interactive prediction using Streamlit

**Important:** This project is for educational purposes only. It is not a medical diagnostic tool and must not be used to make healthcare decisions.

## 2. Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Git and GitHub

## 3. Dataset

The project uses the Pima Indians Diabetes dataset.

- Records: 768
- Features: 8
- Target: Outcome
- Outcome 0: Non-diabetic class
- Outcome 1: Diabetic class

The dataset contains the following features:

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | 2-hour serum insulin |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age in years |
| Outcome | Target variable |

## 4. Project Structure

```text
Diabetes_Prediction/
├── data/
│   └── diabetes.csv
├── models/
│   └── best_diabetes_model.pkl
├── reports/
│   ├── outcome_distribution.png
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── decision_tree_confusion_matrix.png
│   ├── decision_tree_roc_curve.png
│   └── decision_tree_feature_importance.png
├── src/
│   ├── 01_data_understanding.py
│   ├── 02_exploratory_data_analysis.py
│   ├── 03_data_preprocessing.py
│   ├── 04_logistic_regression.py
│   ├── 05_decision_tree.py
│   ├── 06_random_forest.py
│   ├── 07_model_evaluation.py
│   ├── 08_decision_tree_tuning.py
│   ├── 09_save_model.py
│   ├── 10_test_saved_model.py
│   └── app.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 5. Data Preprocessing

The dataset was inspected for missing values, duplicate records and invalid zero values.

Zero values in the following columns were treated as missing measurements:
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI

Median imputation was used to handle missing values.

The dataset was split into:
- Training data: 80%
- Testing data: 20%

Stratified sampling was used to maintain the target class distribution.

