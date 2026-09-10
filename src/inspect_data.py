import pandas as pd

# Load the dataset
file_path = "data/customer_churn_sample_500.csv"
df = pd.read_csv(file_path)

# Display basic information
print("CUSTOMER CHURN DATASET")
print("-" * 40)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nChurn distribution:")
print(df["churn"].value_counts())