# Customer Churn Prediction

This project is a web application that predicts customer churn using a RandomForest model. The application can be deployed on Heroku or run locally on a Windows environment. The application also includes features for visualizing data and evaluating the model.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Running Locally](#running-locally)
- [Deploying to Heroku](#deploying-to-heroku)
- [Model Retraining](#model-retraining)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Features

- Predict customer churn based on input data.
- Visualize key metrics such as tenure distribution, monthly charges, and correlation matrix.
- Evaluate the model on test data.
- Retrain the model with new data.
- Configured for deployment on Heroku and local Windows environments.

## Installation

### Requirements

- Python 3.8+
- `virtualenv` for setting up a virtual environment (optional but recommended)

### Installation Steps

1. Clone the repository:
    ```
    git clone https://github.com/Emmcrispy/churn_prediction.git
    cd churn_prediction
    ```

2. Create and activate a virtual environment:
    ```
    virtualenv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install dependencies:
    - For Windows:
        ```
        pip install -r requirements-windows.txt
        ```
    - For other platforms:
        ```
        pip install -r requirements.txt
        ```

## Running Locally

### Start the Application

To run the application locally, use the following command:
   ```
   python app.py
   ```

## Access the Application
 - Navigate to http://127.0.0.1:5000 in your web browser.

 Important Note
 - When running in a local or virtual environment, ensure that the application uses http instead of https to avoid SSL-related issues.

 ## Deploying to Heroku

## Prerequisites
 - Heroku CLI installed and logged in.

 ## Deployment Steps

1. Create a new Heroku application:
```
heroku create your-app-name
```

2. Set the buildpack to Python:
```
heroku buildpacks:set heroku/python
```

3. Push the code to Heroku:
```
git push heroku main
```

4. Set environment variables on Heroku if needed:
```
heroku config:set DATABASE_URL=your_database_url
heroku config:set SECRET_KEY=your_secret_key
```

5. Open the application:
```
heroku open
```

## Model Retraining
To retrain the model with new data:

1. Upload a CSV file using the retraining form on the home page.
2. The model will be retrained and saved.

## Evaluation
To evaluate the model:

1. Click on the "Evaluate Model" button on the home page.
2. The evaluation metrics (Accuracy, Precision, Recall, F1 Score) will be displayed.

## Contributing
Feel free to fork this repository, create a new branch, and submit a pull request with your changes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.