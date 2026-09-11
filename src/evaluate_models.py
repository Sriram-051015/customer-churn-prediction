import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# Load dataset
file_path = "data/customer_churn_features.csv"
df = pd.read_csv(file_path)

print("MODEL EVALUATION")
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


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Logistic Regression pipeline
logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000))
    ]
)

logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)


print("\nLOGISTIC REGRESSION")
print("-" * 30)

print(f"Accuracy:  {accuracy_score(y_test, logistic_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y_test, logistic_pred) * 100:.2f}%")
print(f"Recall:    {recall_score(y_test, logistic_pred) * 100:.2f}%")
print(f"F1-score:  {f1_score(y_test, logistic_pred) * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, logistic_pred))


# Random Forest pipeline
random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                max_depth=10
            )
        )
    ]
)

random_forest_model.fit(X_train, y_train)

random_forest_pred = random_forest_model.predict(X_test)


print("\nRANDOM FOREST")
print("-" * 30)

print(f"Accuracy:  {accuracy_score(y_test, random_forest_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y_test, random_forest_pred) * 100:.2f}%")
print(f"Recall:    {recall_score(y_test, random_forest_pred) * 100:.2f}%")
print(f"F1-score:  {f1_score(y_test, random_forest_pred) * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, random_forest_pred))


print("\nModel evaluation completed successfully!")