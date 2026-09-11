import pandas as pd
import joblib


# Load trained model
model_path = "models/logistic_regression_model.pkl"
model = joblib.load(model_path)

print("CUSTOMER CHURN PREDICTION")
print("-" * 50)


# Create sample customer
customer = pd.DataFrame([
    {
        "age": 25,
        "gender": "Male",
        "city": "Chennai",
        "tenure_months": 8,
        "monthly_charges": 85.0,
        "total_charges": 680.0,
        "contract_type": "Monthly",
        "payment_method": "Credit Card",
        "internet_service": "Fiber",
        "support_calls": 6,
        "complaints": 3,
        "login_frequency": 10,
        "subscription_type": "Basic",
        "satisfaction_score": 4,
        "inactive_days": 40,
        "tenure_years": 8 / 12,
        "calculated_monthly_charge": 680.0 / 8,
        "support_complaint_ratio": 3 / 6
    }
])


# Make prediction
prediction = model.predict(customer)[0]

# Get churn probability
probability = model.predict_proba(customer)[0][1]


# Display result
if prediction == 1:
    result = "Yes"
else:
    result = "No"


print("\nCustomer Details:")
print(customer.to_string(index=False))

print("\nChurn Prediction:")
print(f"Will the customer churn? {result}")

print(f"Churn Probability: {probability * 100:.2f}%")

print("\nPrediction completed successfully!")