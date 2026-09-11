import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load feature-engineered dataset
file_path = "data/customer_churn_features.csv"
df = pd.read_csv(file_path)

print("RANDOM FOREST CUSTOMER CHURN MODEL")
print("-" * 50)

# Separate input features and target
X = df.drop(columns=["customer_id", "churn", "churn_encoded"])
y = df["churn_encoded"]

# Identify categorical columns
categorical_columns = X.select_dtypes(include=["str"]).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)

# Transform categorical features
X_processed = preprocessor.fit_transform(X)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=10
)

# Train model
model.fit(X_train, y_train)

print("\nRandom Forest training completed!")

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nRandom Forest Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nRandom Forest model completed successfully!")