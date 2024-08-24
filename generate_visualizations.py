import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned data
df = pd.read_csv('data/processed/cleaned_churn_data.csv')
print("Data loaded successfully.")

# Check for missing or incorrect column names
print("Columns in DataFrame:", df.columns.tolist())

# Check for NaN values
print("NaN values in each column:\n", df.isna().sum())

# 1. Tenure Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['tenure'], bins=30, kde=True)
plt.title('Tenure Distribution')
plt.xlabel('Tenure (months)')
plt.ylabel('Frequency')
plt.savefig('static/images/tenure_distribution.png')
plt.close()
print("Tenure distribution saved.")

# 2. Correlation Matrix
try:
    plt.figure(figsize=(12, 10))
    # Exclude non-numeric columns
    corr = df.drop(columns=['customerID']).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('static/images/correlation_matrix.png')
    plt.close()
    print("Correlation matrix saved.")
except Exception as e:
    print(f"Error generating correlation matrix: {e}")

# 3. Monthly Charges Boxplot by Churn Status
try:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churn_Yes', y='MonthlyCharges', data=df)
    plt.title('Monthly Charges by Churn Status')
    plt.xlabel('Churn (Yes = 1, No = 0)')
    plt.ylabel('Monthly Charges ($)')
    plt.savefig('static/images/monthly_charges_boxplot.png')
    plt.close()
    print("Monthly charges boxplot saved.")
except Exception as e:
    print(f"Error generating monthly charges boxplot: {e}")
