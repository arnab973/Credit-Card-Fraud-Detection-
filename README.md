# Credit Card Fraud Detection

This is a Flask-based web application for detecting credit card fraud using machine learning.

## Features

- Web interface for inputting transaction details
- Machine learning model for fraud prediction
- REST API endpoints for integration

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/arnab973/Credit-Card-Fraud-Detection-.git
   cd Credit-Card-Fraud-Detection-
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To start the application, run:
```
python app.py
```

Or using Python launcher:
```
py app.py
```

The application will be available at http://localhost:5000

## API Endpoints

- `GET /`: Home page
- `GET /generate_features`: Generate random features for V1-V28
- `POST /predict`: Predict fraud based on transaction data

## Model

The model is trained using RandomForestClassifier on the credit card dataset with SMOTE for handling imbalance.

## License

MIT License
