# Customer Churn Prediction

A machine learning project that predicts whether a customer is likely to churn based on customer behavior, subscription details, service usage, and satisfaction-related features.

## Project Overview

Customer churn occurs when a customer stops using a company's products or services.

The goal of this project is to analyze customer data, identify important churn patterns, build machine learning models, compare their performance, and create a prediction system for new customers.

The project also includes a Power BI dashboard for visual analysis of customer churn patterns.

## Objectives

* Inspect and clean customer data
* Perform exploratory data analysis
* Identify customer churn patterns
* Create useful features
* Prepare data for machine learning
* Train classification models
* Evaluate model performance
* Save the final trained model
* Predict churn for new customers
* Create a Power BI dashboard for customer churn analysis

## Dataset

The dataset contains 500 customer records and 17 original columns.

Important features include:

* Age
* Gender
* City
* Tenure
* Monthly charges
* Total charges
* Contract type
* Payment method
* Internet service
* Support calls
* Complaints
* Login frequency
* Subscription type
* Satisfaction score
* Inactive days
* Churn

### Target Variable

* **Churn = Yes** - Customer is likely to leave the service
* **Churn = No** - Customer is likely to stay with the service

## Project Workflow

```text
Raw Customer Data
        |
        v
Data Inspection
        |
        v
Data Cleaning
        |
        v
Exploratory Data Analysis
        |
        v
Feature Engineering
        |
        v
Data Preparation
        |
        v
Model Training
        |
        v
Model Evaluation
        |
        v
Final Model Selection
        |
        v
Model Saving
        |
        v
Customer Churn Prediction
        |
        v
Power BI Dashboard
```

## Machine Learning Models

Two classification models were developed and evaluated:

* Logistic Regression
* Random Forest

## Model Results

| Model               | Accuracy | Precision | Recall | F1-score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |   84.00% |    85.07% | 90.48% |   87.69% |
| Random Forest       |   79.00% |    79.17% | 90.48% |   84.44% |

Logistic Regression achieved the higher overall performance on the test dataset and was selected as the final model for customer churn prediction.

### Confusion Matrix

![Logistic Regression Confusion Matrix](results/logistic_confusion_matrix.png)

### Feature Importance

![Logistic Regression Feature Importance](results/logistic_feature_importance.png)

### Classification Report

The classification report provides detailed performance metrics for both churn classes.

| Class    | Precision | Recall | F1-score |
| -------- | --------: | -----: | -------: |
| No Churn |      0.82 |   0.73 |     0.77 |
| Churn    |      0.85 |   0.90 |     0.88 |

The model achieved an overall accuracy of **84%** on the test dataset.

## Power BI Dashboard

A Power BI dashboard was created to provide visual analysis of customer churn.

The dashboard contains two pages:

### Page 1 - Overview

Key performance indicators:

* Total Customers
* Churned Customers
* Churn Rate
* Average Monthly Charges

Chart:

* Churn by Contract Type

### Page 2 - Customer Insights

Charts:

* Churn by Internet Service
* Churn by Subscription Type
* Churn by City
* Churn by Payment Method
* Churn by Age Group

The Power BI dashboard file is available at:

```text
dashboard/customer_churn_dashboard.pbix
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Power BI
* Git & GitHub

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Sriram-051015/customer-churn-prediction.git
```

### 2. Open the project folder

```bash
cd customer-churn-prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install the required libraries

```bash
pip install -r requirements.txt
```

## How to Run the Project

Run the Python files in the following order:

```text
1. inspect_data.py
2. clean_data.py
3. eda.py
4. churn_analysis.py
5. visualize_churn.py
6. feature_engineering.py
7. prepare_data.py
8. baseline_model.py
9. random_forest_model.py
10. evaluate_models.py
11. improved_logistic_model.py
12. predict_churn.py
13. interactive_prediction.py
14. confusion_matrix_plot.py
15. feature_importance.py
16. classification_report.py
```

All Python scripts are located inside the `src` folder.

Example:

```bash
python src/inspect_data.py
```

## Example Prediction

The trained Logistic Regression model can predict whether a new customer is likely to churn and provide an estimated churn probability.

Example:

```text
Will the customer churn? Yes
Churn Probability: 99.85%
```

## Key Churn Insights

The exploratory analysis identified several important patterns:

* Monthly contract customers showed a higher churn rate.
* Customers with higher support calls showed stronger churn tendency.
* Customers with more complaints showed stronger churn tendency.
* Longer-term contracts generally showed lower churn.
* Customer satisfaction and inactivity were important predictive factors.

## Future Improvements

* Try additional machine learning algorithms
* Perform hyperparameter tuning
* Use a larger real-world dataset
* Build a web-based prediction interface
* Deploy the model as an API

## Project Structure

```text
customer_churn_prediction/
|
+-- data/
|   +-- customer_churn_sample_500.csv
|   +-- customer_churn_cleaned.csv
|   +-- customer_churn_features.csv
|
+-- src/
|   +-- inspect_data.py
|   +-- clean_data.py
|   +-- eda.py
|   +-- churn_analysis.py
|   +-- visualize_churn.py
|   +-- feature_engineering.py
|   +-- prepare_data.py
|   +-- baseline_model.py
|   +-- random_forest_model.py
|   +-- evaluate_models.py
|   +-- improved_logistic_model.py
|   +-- predict_churn.py
|   +-- interactive_prediction.py
|   +-- confusion_matrix_plot.py
|   +-- feature_importance.py
|   +-- classification_report.py
|
+-- models/
|   +-- logistic_regression_model.pkl
|
+-- results/
|   +-- eda_summary.txt
|   +-- model_results.txt
|   +-- classification_report.txt
|   +-- logistic_confusion_matrix.png
|   +-- logistic_feature_importance.png
|
+-- dashboard/
|   +-- customer_churn_dashboard.pbix
|
+-- notebooks/
+-- requirements.txt
+-- README.md
+-- .gitignore
```

## Author

Sriram

## Disclaimer

This project is created for educational and portfolio purposes. The dataset is a sample dataset and the predictions should not be treated as real-world business decisions.
