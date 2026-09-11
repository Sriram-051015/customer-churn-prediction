import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load feature-engineered dataset
file_path = "data/customer_churn_features.csv"
df = pd.read_csv(file_path)

print("DATA PREPARATION FOR MACHINE LEARNING")
print("-" * 50)

# Remove customer ID because it is only an identifier
X = df.drop(columns=["customer_id", "churn", "churn_encoded"])
y = df["churn_encoded"]

# Identify categorical columns

categorical_columns = X.select_dtypes(include=["str"]).columns.tolist()


# Identify numerical columns
numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)

# Create preprocessing transformer
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_columns
        )
    ],
    remainder="passthrough"
)

# Transform the data
X_processed = preprocessor.fit_transform(X)

# Get feature names
encoded_columns = preprocessor.named_transformers_[
    "categorical"
].get_feature_names_out(categorical_columns)

feature_names = list(encoded_columns) + numerical_columns

# Convert processed data into DataFrame
X_processed_df = pd.DataFrame(
    X_processed,
    columns=feature_names
)

print("\nOriginal feature count:")
print(X.shape[1])

print("\nProcessed feature count:")
print(X_processed_df.shape[1])

print("\nProcessed data shape:")
print(X_processed_df.shape)

print("\nFirst 5 rows of processed data:")
print(X_processed_df.head())

print("\nTarget distribution:")
print(y.value_counts())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_processed_df,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nData preparation completed successfully!")