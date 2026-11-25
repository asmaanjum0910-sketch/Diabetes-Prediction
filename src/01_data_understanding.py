
import pandas as pd
from pathlib import Path

# Find the project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset location
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# 1. Display first five records
print("\n--- FIRST FIVE RECORDS ---")
print(df.head())

# 2. Display dataset dimensions
print("\n--- DATASET SHAPE ---")
print(df.shape)

# 3. Display column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# 4. Display data types
print("\n--- DATA TYPES ---")
print(df.dtypes)

# 5. Display dataset information
print("\n--- DATASET INFORMATION ---")
df.info()

# 6. Display statistical summary
print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())

# 7. Check missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# 8. Check duplicate records
print("\n--- DUPLICATE RECORDS ---")
print(df.duplicated().sum())


# 9. Investigate zero values
print("\n--- ZERO VALUE COUNTS ---")

zero_columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for column in zero_columns:
    zero_count = (df[column] == 0).sum()
    percentage = (zero_count / len(df)) * 100

    print(
        f"{column}: {zero_count} zeros "
        f"({percentage:.2f}%)"
    )


# 10. Target variable distribution
print("\n--- OUTCOME DISTRIBUTION ---")
print(df["Outcome"].value_counts())

print("\n--- OUTCOME PERCENTAGES ---")
print(
    df["Outcome"].value_counts(normalize=True) * 100
)
