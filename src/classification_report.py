import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# Load dataset
file_path = "data/customer_churn_features.csv"
df = pd.read_csv(file_path)

print("LOGISTIC REGRESSION CLASSIFICATION REPORT")
print("-" * 50)


# Separate features and target
X = df.drop(columns=["customer_id", "churn", "churn_encoded"])
y = df["churn_encoded"]


# Identify columns
categorical_columns = X.select_dtypes(include=["str"]).columns.tolist()
numerical_columns = X.select_dtypes(exclude=["str"]).columns.tolist()


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_columns
        )
    ]
)


# Create model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000))
    ]
)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train model
model.fit(X_train, y_train)

print("\nModel training completed!")


# Predictions
y_pred = model.predict(X_test)


# Classification report
report = classification_report(
    y_test,
    y_pred,
    target_names=["No Churn", "Churn"]
)

print("\nClassification Report")
print("-" * 30)
print(report)


# Save report
output_path = "results/classification_report.txt"

with open(output_path, "w") as file:
    file.write("LOGISTIC REGRESSION CLASSIFICATION REPORT\n")
    file.write("=" * 50 + "\n\n")
    file.write(report)

print("Classification report saved successfully!")
print(f"File: {output_path}")