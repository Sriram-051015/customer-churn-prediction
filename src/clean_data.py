import pandas as pd

# Load the dataset
file_path = "data/customer_churn_sample_500.csv"
df = pd.read_csv(file_path)

print("BEFORE CLEANING")
print("-" * 40)
print("Missing values:")
print(df.isnull().sum())

# Fill missing internet service values
df["internet_service"] = df["internet_service"].fillna("No Internet")

# Check missing values after cleaning
print("\nAFTER CLEANING")
print("-" * 40)
print("Missing values:")
print(df.isnull().sum())

# Save the cleaned dataset
output_path = "data/customer_churn_cleaned.csv"
df.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")
print(f"File: {output_path}")