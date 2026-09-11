import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# Load dataset
file_path = "data/customer_churn_features.csv"
df = pd.read_csv(file_path)

print("LOGISTIC REGRESSION FEATURE IMPORTANCE")
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


# Get feature names
encoded_features = model.named_steps[
    "preprocessor"
].named_transformers_[
    "categorical"
].get_feature_names_out(categorical_columns)

feature_names = list(encoded_features) + numerical_columns


# Get logistic regression coefficients
coefficients = model.named_steps[
    "classifier"
].coef_[0]


# Create feature importance dataframe
importance_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
})

importance_df["absolute_coefficient"] = (
    importance_df["coefficient"].abs()
)

importance_df = importance_df.sort_values(
    "absolute_coefficient",
    ascending=False
)


# Display top features
print("\nTop 10 influential features:")
print(
    importance_df[
        ["feature", "coefficient"]
    ].head(10).to_string(index=False)
)


# Select top 10 features
top_features = importance_df.head(10).sort_values(
    "coefficient"
)


# Create visualization
plt.figure(figsize=(10, 6))

plt.barh(
    top_features["feature"],
    top_features["coefficient"]
)

plt.axvline(
    0,
    linewidth=1
)

plt.title("Top 10 Logistic Regression Features")
plt.xlabel("Model Coefficient")
plt.ylabel("Feature")

plt.tight_layout()


# Save visualization
output_path = "results/logistic_feature_importance.png"

plt.savefig(
    output_path,
    dpi=300
)

print("\nFeature importance visualization saved successfully!")
print(f"File: {output_path}")