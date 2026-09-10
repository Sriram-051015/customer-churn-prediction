import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
file_path = "data/customer_churn_cleaned.csv"
df = pd.read_csv(file_path)

# Create results folder if needed
import os
os.makedirs("results", exist_ok=True)

# 1. Churn distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("results/churn_distribution.png")
plt.show()


# 2. Churn by contract type
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="contract_type", hue="churn")
plt.title("Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("results/churn_by_contract.png")
plt.show()


# 3. Churn by subscription type
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="subscription_type", hue="churn")
plt.title("Churn by Subscription Type")
plt.xlabel("Subscription Type")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("results/churn_by_subscription.png")
plt.show()


# 4. Satisfaction score vs churn
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="churn", y="satisfaction_score")
plt.title("Satisfaction Score vs Churn")
plt.xlabel("Churn")
plt.ylabel("Satisfaction Score")
plt.tight_layout()
plt.savefig("results/satisfaction_vs_churn.png")
plt.show()


# 5. Inactive days vs churn
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="churn", y="inactive_days")
plt.title("Inactive Days vs Churn")
plt.xlabel("Churn")
plt.ylabel("Inactive Days")
plt.tight_layout()
plt.savefig("results/inactive_days_vs_churn.png")
plt.show()


# 6. Monthly charges vs churn
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="churn", y="monthly_charges")
plt.title("Monthly Charges vs Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig("results/monthly_charges_vs_churn.png")
plt.show()


print("All visualizations created successfully!")
print("Charts saved in the results folder.")