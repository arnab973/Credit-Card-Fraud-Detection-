# Credit Card Fraud Detection System

This is a modern Flask-based web application and Machine Learning pipeline for detecting credit card fraud using Random Forest Classification and SMOTE oversampling.
For live Demo - https://credit-card-fraud-detection-lilac.vercel.app/

## Features

- **Modern Glassmorphism Dashboard**: Sleek dark-themed UI with interactive risk meters, status indicators, and probability scoring.
- **Quick Test Presets**: Built-in sample buttons (`Safe Sample` & `Fraud Sample`) extracted from authentic `creditcard.csv` transactions for instant model validation.
- **Random Feature Generator**: Synthetic Gaussian noise generator for PCA features (V1–V28).
- **Fraud Risk Assessment**: Computes class predictions, fraud/safe probability percentages, and risk levels (`Low Risk`, `Moderate Risk`, `High Risk`).
- **REST API Endpoints**: Modular backend routes for external application integration.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/arnab973/Credit-Card-Fraud-Detection-.git
   cd Credit-Card-Fraud-Detection-
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Start the Flask application server:
```bash
python app.py
```

The application will be accessible at: `http://localhost:5000`

## API Endpoints

- `GET /` : Renders the web interface dashboard.
- `GET /sample_data/<type>` : Returns sample transaction feature vectors (`safe` or `fraud`).
- `GET /generate_features` : Generates random Gaussian features for V1–V28.
- `POST /predict` : Expects JSON payload with `Time`, `V1`..`V28`, and `Amount`. Returns prediction class, probability percentages, and risk level.

## Model Details

- **Dataset**: Kaggle Credit Card Fraud Detection dataset (284,807 transactions).
- **Algorithm**: `RandomForestClassifier` trained with `SMOTE` oversampling to address severe class imbalance.

## License

MIT License
