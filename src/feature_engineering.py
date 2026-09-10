import pandas as pd

# Load cleaned dataset
file_path = "data/customer_churn_cleaned.csv"
df = pd.read_csv(file_path)

print("FEATURE ENGINEERING")
print("-" * 50)

# Create tenure in years
df["tenure_years"] = df["tenure_months"] / 12

# Calculate average monthly charge based on total charges
df["calculated_monthly_charge"] = (
    df["total_charges"] / df["tenure_months"].replace(0, 1)
)

# Create support complaint ratio
df["support_complaint_ratio"] = (
    df["complaints"] / df["support_calls"].replace(0, 1)
)

# Convert churn from Yes/No to 1/0
df["churn_encoded"] = df["churn"].map({
    "No": 0,
    "Yes": 1
})

print("\nNew features created:")
print("- tenure_years")
print("- calculated_monthly_charge")
print("- support_complaint_ratio")
print("- churn_encoded")

print("\nSample of engineered features:")
print(
    df[
        [
            "customer_id",
            "tenure_months",
            "tenure_years",
            "total_charges",
            "calculated_monthly_charge",
            "support_calls",
            "complaints",
            "support_complaint_ratio",
            "churn",
            "churn_encoded"
        ]
    ].head()
)

print("\nChurn encoding:")
print(df["churn_encoded"].value_counts())

# Save feature-engineered dataset
output_path = "data/customer_churn_features.csv"
df.to_csv(output_path, index=False)

print("\nFeature-engineered dataset saved successfully!")
print(f"File: {output_path}")