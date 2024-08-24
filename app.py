import os
import logging
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from flask import Flask, request, render_template, redirect
from sklearn.ensemble import RandomForestClassifier
from flask import request, redirect, url_for
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from flask import Flask, render_template, request, redirect, url_for, Response
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming the model is a RandomForestClassifier and was trained with certain features
model = RandomForestClassifier(n_estimators=100, random_state=42)

def retrain_model(new_data):
    # Load new data
    df_new = pd.read_csv(new_data)

    # Example: Define the features and target as done previously
    X_new = df_new.drop('Churn_Yes', axis=1)
    y_new = df_new['Churn_Yes']

    # Preprocess and retrain the model
    model.fit(X_new, y_new)

    # Save the retrained model
    model_path = os.path.join('models', 'churn_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print("Model retrained and saved successfully.")
    
def evaluate_model_on_test_data():
    # Load the test data
    test_df = pd.read_csv('/data/processed/cleaned_churn_data.csv')  # Ensure this is your test dataset    
    X_test = test_df.drop('Churn_Yes', axis=1)
    y_test = test_df['Churn_Yes']

    # Align the test data with the model's expected input columns
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_test = X_test.reindex(columns=df.drop('Churn_Yes', axis=1).columns, fill_value=0)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return metrics

# Generate live visualizations    
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

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
database_url = os.getenv('DATABASE_URL')
secret_key = os.getenv('SECRET_KEY')

app.config['SECRET_KEY'] = secret_key

# Load the trained model
model_path = os.path.join('models', 'churn_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load cleaned data for feature alignment
df = pd.read_csv('data/processed/cleaned_churn_data.csv')

# Function to validate input data
def validate_input(input_dict):
    try:
        # Convert and validate numerical fields
        input_dict['tenure'] = int(input_dict['tenure'])
        input_dict['MonthlyCharges'] = float(input_dict['MonthlyCharges'])

        # Remove non-numeric fields like 'customerID'
        input_dict.pop('customerID', None)

        # Add similar validations for other fields if necessary
        if input_dict['tenure'] < 0:
            raise ValueError("Tenure must be a positive integer.")
        if input_dict['MonthlyCharges'] < 0:
            raise ValueError("Monthly Charges must be a positive number.")

        # Validation passed
        return True, input_dict
    except ValueError as ve:
        # Log the validation error
        logging.error(f"Input validation error: {ve}")
        return False, str(ve)

# Force HTTPS
@app.before_request
def before_request():
    if not app.debug and request.url.startswith('http://'):
        return redirect(request.url.replace('http://', 'https://', 1))

@app.route('/retrain', methods=['POST'])
def retrain():
    # Assuming the new data is uploaded as a file
    if 'new_data_file' in request.files:
        file = request.files['new_data_file']
        filepath = os.path.join('data/processed/', file.filename)
        file.save(filepath)
        
        # Retrain the model with the new data
        retrain_model(filepath)
        
        return redirect(url_for('home'))
    return "No file uploaded", 400

@app.route('/evaluate')
def evaluate():
    try:
        # Log the evaluation start
        logging.info("Starting model evaluation...")

        metrics = evaluate_model_on_test_data()

        # Log successful evaluation
        logging.info(f"Model evaluation completed successfully. Metrics: {metrics}")

        return render_template('evaluation.html', metrics=metrics)
    except FileNotFoundError as e:
        # Handle specific file not found error
        logging.error(f"Test data file not found: {e}")
        return render_template('index.html', prediction_text="Test data file not found. Please check the file path.")
    except Exception as e:
        # Handle any other exceptions that may occur
        logging.error(f"Error during evaluation: {e}")
        return render_template('index.html', prediction_text="An error occurred during evaluation.")

def evaluate_model_on_test_data():
    # Update the path to the correct location of your CSV file
    test_data_path = 'data/processed/cleaned_churn_data.csv'  # Use the correct path to your CSV file

    # Load the test data
    test_df = pd.read_csv(test_data_path)
    
    X_test = test_df.drop('Churn_Yes', axis=1)
    y_test = test_df['Churn_Yes']

    # Align the test data with the model's expected input columns
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_test = X_test.reindex(columns=df.drop('Churn_Yes', axis=1).columns, fill_value=0)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return metrics

# Route to update visualizations
@app.route('/update_visualizations')
def update_visualizations():
    plot_url_tenure, plot_url_monthly_charges, plot_url_correlation_matrix = generate_visualizations()
    return render_template('index.html', plot_url_tenure=plot_url_tenure, plot_url_monthly_charges=plot_url_monthly_charges, plot_url_correlation_matrix=plot_url_correlation_matrix)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the preprocessing pipeline
def preprocess_data(df):
    # Define numerical and categorical columns
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                            'MultipleLines', 'InternetService', 'OnlineSecurity', 
                            'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                            'StreamingTV', 'StreamingMovies', 'Contract', 
                            'PaperlessBilling', 'PaymentMethod']
    
    # Create transformers for numerical and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Fill missing values
        ('scaler', StandardScaler())  # Scale numerical values
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical values
    ])

    # Combine transformers into a single column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Apply preprocessing
    df_processed = preprocessor.fit_transform(df)
    
    # Return the preprocessed DataFrame
    return pd.DataFrame(df_processed, columns=preprocessor.get_feature_names_out())

# Route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form.to_dict()
        logging.debug(f"Received input data: {input_data}")

        is_valid, validated_data = validate_input(input_data)
        if not is_valid:
            return render_template('index.html', prediction_text=f"Validation Error: {validated_data}")

        # Convert validated data into a DataFrame
        input_df = pd.DataFrame([validated_data])

        # Ensure 'customerID' is removed from the data
        input_df = input_df.drop(columns=['customerID'], errors='ignore')

        # Convert categorical variables to dummy variables
        input_df = pd.get_dummies(input_df, drop_first=True)

        # Align the input dataframe with the model's expected input columns
        input_df = input_df.reindex(columns=df.drop('Churn_Yes', axis=1).columns, fill_value=0)

        # Predict
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]

        prediction_text = 'Yes' if prediction == 1 else 'No'

        result = f"Customer Churn Prediction: {'Yes' if prediction == 1 else 'No'} (Probability: {prediction_proba:.2f})"

        # Generate visualizations
        plot_url_tenure, plot_url_monthly_charges, plot_url_correlation_matrix = generate_visualizations(df)

        return render_template('index.html', prediction_text=result,
                        plot_url_tenure=plot_url_tenure,
                        plot_url_monthly_charges=plot_url_monthly_charges,
                        plot_url_correlation_matrix=plot_url_correlation_matrix)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return render_template('index.html', prediction_text="An error occurred during prediction.")

if __name__ == '__main__':
    app.run(debug=True)
