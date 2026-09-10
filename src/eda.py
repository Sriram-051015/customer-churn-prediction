import pandas as pd

# Load the cleaned dataset
file_path = "data/customer_churn_cleaned.csv"
df = pd.read_csv(file_path)

print("CUSTOMER CHURN - EXPLORATORY DATA ANALYSIS")
print("-" * 50)

# Basic dataset information
print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

# Churn distribution
print("\nChurn distribution:")
print(df["churn"].value_counts())

print("\nChurn percentage:")
print(df["churn"].value_counts(normalize=True) * 100)

# Numerical columns summary
print("\nNumerical columns summary:")
print(df.describe())

# Categorical columns summary
print("\nContract type distribution:")
print(df["contract_type"].value_counts())

print("\nSubscription type distribution:")
print(df["subscription_type"].value_counts())

print("\nInternet service distribution:")
print(df["internet_service"].value_counts())

# Average values based on churn
print("\nAverage values by churn status:")
print(
    df.groupby("churn")[
        [
            "age",
            "tenure_months",
            "monthly_charges",
            "support_calls",
            "complaints",
            "login_frequency",
            "satisfaction_score",
            "inactive_days"
        ]
    ].mean()
)