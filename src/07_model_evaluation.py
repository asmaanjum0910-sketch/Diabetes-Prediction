
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score
)

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

# Build Decision Tree pipeline
model = Pipeline(
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
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# Create reports folder if needed
reports_dir = BASE_DIR / "reports"
reports_dir.mkdir(exist_ok=True)

# 1. Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Decision Tree - Confusion Matrix")
plt.tight_layout()
plt.savefig(reports_dir / "decision_tree_confusion_matrix.png")
plt.show()

# 2. ROC Curve
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("Decision Tree - ROC Curve")
plt.tight_layout()
plt.savefig(reports_dir / "decision_tree_roc_curve.png")
plt.show()

# 3. Feature Importance
importance = model.named_steps["classifier"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance_df)

plt.figure(figsize=(9, 5))
plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)
plt.gca().invert_yaxis()
plt.title("Decision Tree - Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(reports_dir / "decision_tree_feature_importance.png")
plt.show()

print("\nVisualisations saved successfully!")
