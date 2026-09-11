import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load dataset
file_path = "data/customer_churn_features.csv"
df = pd.read_csv(file_path)

print("IMPROVED LOGISTIC REGRESSION MODEL")
print("-" * 50)


# Separate features and target
X = df.drop(columns=["customer_id", "churn", "churn_encoded"])
y = df["churn_encoded"]


# Identify columns
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()


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


# Create pipeline
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


print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)


# Train model
model.fit(X_train, y_train)

print("\nImproved Logistic Regression training completed!")


# Predictions
y_pred = model.predict(X_test)


# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("\nIMPROVED MODEL RESULTS")
print("-" * 30)
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-score:  {f1 * 100:.2f}%")


# Save trained model
model_path = "models/logistic_regression_model.pkl"

joblib.dump(model, model_path)

print("\nTrained model saved successfully!")
print(f"Model file: {model_path}")

print("\nStep 30 completed successfully!")