import os
import logging
import pandas as pd
import pickle
from flask import Flask, request, render_template, redirect
from sklearn.ensemble import RandomForestClassifier
from flask import request, redirect, url_for


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

        # Add similar validations for other fields
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
    if request.url.startswith('http://'):
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

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from the form
        input_data = request.form.to_dict()

        # Validate inputs
        is_valid, validated_data = validate_input(input_data)
        if not is_valid:
            return render_template('index.html', prediction_text=f"Validation Error: {validated_data}")

        # Process validated input data
        input_df = pd.DataFrame([validated_data])

        # Handle categorical variables using one-hot encoding
        input_df = pd.get_dummies(input_df, drop_first=True)

        # Align input features with training features
        input_df = input_df.reindex(columns=df.drop('Churn', axis=1).columns, fill_value=0)

        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]  # Probability of churn

        # Prepare the result
        result = f"Customer Churn Prediction: {'Yes' if prediction == 1 else 'No'} (Probability: {prediction_proba:.2f})"

        logging.info("Prediction successful.")
        return render_template('index.html', prediction_text=result)
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return render_template('index.html', prediction_text="An error occurred during prediction.")

if __name__ == '__main__':
    app.run(debug=True)
