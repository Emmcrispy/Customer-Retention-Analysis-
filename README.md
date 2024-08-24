# Customer Churn Prediction Application

This project is a web-based application designed to predict customer churn using machine learning. The application provides a user-friendly interface for making predictions, visualizing data, and retraining the predictive model with new customer data.

## Features

- **Predict Customer Churn:** Input customer data to predict whether they are likely to churn.
- **Model Retraining:** Upload new datasets to retrain the machine learning model.
- **Data Visualizations:** View key visualizations such as tenure distribution, correlation matrix, and monthly charges by churn status.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Installation

Follow these steps to set up the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Set Up a Virtual Environment

Create and activate a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install the necessary Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Your Data

Ensure your dataset is in CSV format with the following columns: `tenure`, `MonthlyCharges`, `Churn`, and other relevant features.

### 2. Running the Application

Start the Flask web server to run the application locally.

```bash
python app.py
```

Access the application in your web browser at `http://127.0.0.1:5000/`.

### 3. Predict Customer Churn

Use the web interface to input customer data and predict whether they will churn.

### 4. Retrain the Model

Upload a new CSV dataset to retrain the model. Navigate to the "Upload New Data for Model Retraining" section, upload the file, and click "Upload and Retrain Model".

### 5. View Data Visualizations

View visualizations such as the tenure distribution, correlation matrix, and monthly charges by churn status directly on the web interface.

## Project Structure

```plaintext
customer-churn-prediction/
├── app.py                       # Main Flask application
├── data/
│   ├── raw/                     # Raw data files
│   └── processed/               # Processed data files
├── models/
│   └── churn_model.pkl          # Saved machine learning model
├── static/
│   ├── css/
│   │   └── styles.css           # CSS styles
│   └── images/                  # Data visualizations
├── templates/
│   └── index.html               # HTML template for the web interface
├── generate_visualizations.py   # Script to generate visualizations
├── data_preparation.py          # Script for data preparation
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Deployment

### Deploying to Heroku

Follow these steps to deploy your application to Heroku.

1. **Login to Heroku:**

   ```bash
   heroku login
   ```

2. **Create a Heroku Application:**

   ```bash
   heroku create your-app-name
   ```

3. **Deploy the Code:**

   ```bash
   git push heroku master
   ```

4. **Set Up a Procfile:**

   Ensure there is a `Procfile` in the root directory with the following content:

   ```plaintext
   web: gunicorn app:app
   ```

5. **Scale the Web Server:**

   ```bash
   heroku ps:scale web=1
   ```

6. **Open Your Application:**

   ```bash
   heroku open
   ```

### Environment Variables

Use the Heroku dashboard or CLI to set any necessary environment variables (e.g., `SECRET_KEY`, `DATABASE_URL`).

## Contributing

Contributions are welcome! Fork the repository and create a pull request to contribute.

## License

This project is licensed under a proprietary license. Unauthorized copying, distribution, modification, or use of this software is strictly prohibited without the explicit permission of the author.

For more details, see the [LICENSE](LICENSE) file.

