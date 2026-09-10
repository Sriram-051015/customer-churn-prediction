import pandas as pd

# Load the cleaned dataset
file_path = "data/customer_churn_cleaned.csv"
df = pd.read_csv(file_path)

print("CUSTOMER CHURN ANALYSIS")
print("-" * 50)

# Churn rate by contract type
print("\nChurn rate by contract type:")
contract_churn = pd.crosstab(
    df["contract_type"],
    df["churn"],
    normalize="index"
) * 100

print(contract_churn.round(2))


# Churn rate by subscription type
print("\nChurn rate by subscription type:")
subscription_churn = pd.crosstab(
    df["subscription_type"],
    df["churn"],
    normalize="index"
) * 100

print(subscription_churn.round(2))


# Churn rate by internet service
print("\nChurn rate by internet service:")
internet_churn = pd.crosstab(
    df["internet_service"],
    df["churn"],
    normalize="index"
) * 100

print(internet_churn.round(2))


# Churn rate by gender
print("\nChurn rate by gender:")
gender_churn = pd.crosstab(
    df["gender"],
    df["churn"],
    normalize="index"
) * 100

print(gender_churn.round(2))


# Average customer characteristics by churn
print("\nAverage customer characteristics by churn:")
average_values = df.groupby("churn")[
    [
        "tenure_months",
        "monthly_charges",
        "support_calls",
        "complaints",
        "login_frequency",
        "satisfaction_score",
        "inactive_days"
    ]
].mean()

print(average_values.round(2))


print("\nAnalysis completed successfully!")
