import pandas as pd

# Load the dataset
df = pd.read_csv('data/raw/customer_churn.csv')

# Convert TotalCharges to numeric, coercing errors to NaN, then fill with median or mean
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Ensure 'Churn' is correctly referenced for one-hot encoding
df = pd.get_dummies(df, columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                                 'PaperlessBilling', 'PaymentMethod', 'Churn'], drop_first=True)

# Save cleaned data
df.to_csv('data/processed/cleaned_churn_data.csv', index=False)
