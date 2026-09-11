import pandas as pd
import joblib


# Load trained model
model_path = "models/logistic_regression_model.pkl"
model = joblib.load(model_path)

print("CUSTOMER CHURN PREDICTION SYSTEM")
print("-" * 50)

print("\nEnter customer details:")

age = int(input("Age: "))
gender = input("Gender (Male/Female): ")
city = input("City: ")
tenure_months = int(input("Tenure in months: "))
monthly_charges = float(input("Monthly charges: "))
total_charges = float(input("Total charges: "))
contract_type = input("Contract type (Monthly/Yearly/2-Year): ")
payment_method = input("Payment method: ")
internet_service = input("Internet service (DSL/Fiber/No Internet): ")
support_calls = int(input("Support calls: "))
complaints = int(input("Complaints: "))
login_frequency = int(input("Login frequency: "))
subscription_type = input("Subscription type (Basic/Standard/Premium): ")
satisfaction_score = int(input("Satisfaction score (1-10): "))
inactive_days = int(input("Inactive days: "))


# Feature engineering
tenure_years = tenure_months / 12

calculated_monthly_charge = (
    total_charges / tenure_months
    if tenure_months != 0
    else 0
)

support_complaint_ratio = (
    complaints / support_calls
    if support_calls != 0
    else 0
)


# Create customer dataframe
customer = pd.DataFrame([
    {
        "age": age,
        "gender": gender,
        "city": city,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "support_calls": support_calls,
        "complaints": complaints,
        "login_frequency": login_frequency,
        "subscription_type": subscription_type,
        "satisfaction_score": satisfaction_score,
        "inactive_days": inactive_days,
        "tenure_years": tenure_years,
        "calculated_monthly_charge": calculated_monthly_charge,
        "support_complaint_ratio": support_complaint_ratio
    }
])


# Make prediction
prediction = model.predict(customer)[0]

probability = model.predict_proba(customer)[0][1]


# Convert prediction
if prediction == 1:
    result = "Yes"
else:
    result = "No"


# Display result
print("\n" + "=" * 50)
print("PREDICTION RESULT")
print("=" * 50)

print(f"\nWill the customer churn? {result}")
print(f"Churn probability: {probability * 100:.2f}%")

print("\nPrediction completed successfully!")