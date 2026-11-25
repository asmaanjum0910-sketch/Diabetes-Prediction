
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"

# Reports folder
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Work on a copy for visualization
eda_df = df.copy()

# Replace medically implausible zero placeholders with NaN
zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

eda_df[zero_columns] = eda_df[zero_columns].replace(0, float("nan"))

sns.set_theme(style="whitegrid")


# 1. Outcome distribution
plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Outcome")

plt.title("Diabetes Outcome Distribution")
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Number of Records")

plt.tight_layout()
plt.savefig(REPORT_DIR / "outcome_distribution.png")
plt.show()


# 2. Feature distributions
features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

eda_df[features].hist(
    figsize=(14, 10),
    bins=20,
    edgecolor="black"
)

plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.savefig(REPORT_DIR / "feature_distributions.png")
plt.show()


# 3. Correlation heatmap
plt.figure(figsize=(11, 8))

correlation = eda_df.corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.savefig(REPORT_DIR / "correlation_heatmap.png")
plt.show()


# 4. Display correlation with target
print("\n--- CORRELATION WITH OUTCOME ---")

print(
    correlation["Outcome"]
    .sort_values(ascending=False)
)

print("\nEDA completed successfully!")
