import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def generate_visualizations(df):
    # Tenure Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['tenure'], kde=False, bins=30)
    plt.title('Tenure Distribution')
    plt.xlabel('Tenure (months)')
    plt.ylabel('Frequency')

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url_tenure = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Monthly Charges Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='MonthlyCharges', data=df)
    plt.title('Monthly Charges Boxplot')

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url_monthly_charges = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Remove non-numeric columns for correlation matrix
    numeric_df = df.select_dtypes(include=['number'])

    # Correlation Matrix
    plt.figure(figsize=(10, 6))
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url_correlation_matrix = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return plot_url_tenure, plot_url_monthly_charges, plot_url_correlation_matrix

# Assuming you have a DataFrame `df` already loaded
df = pd.read_csv(r'C:/Users/elman/churn_prediction/data/processed/cleaned_churn_data.csv')
plot_url_tenure, plot_url_monthly_charges, plot_url_correlation_matrix = generate_visualizations(df)

# Save the images to files as part of your test
with open("tenure_distribution.png", "wb") as f:
    f.write(base64.b64decode(plot_url_tenure))

with open("monthly_charges_boxplot.png", "wb") as f:
    f.write(base64.b64decode(plot_url_monthly_charges))

with open("correlation_matrix.png", "wb") as f:
    f.write(base64.b64decode(plot_url_correlation_matrix))

print("Visualizations generated and saved successfully.")

