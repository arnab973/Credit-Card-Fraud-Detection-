from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import warnings

# Suppress version warning for model unpickling
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Feature names in exact order expected by the model
FEATURE_NAMES = ['Time'] + [f'V{i+1}' for i in range(28)] + ['Amount']

# Pre-extracted sample transactions from dataset for instant testing
SAMPLE_TRANSACTIONS = {
    "safe": {
        "Time": 0.0,
        "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155, "V5": -0.338321,
        "V6": 0.462388, "V7": 0.239599, "V8": 0.098698, "V9": 0.363787, "V10": 0.090794,
        "V11": -0.551600, "V12": -0.617801, "V13": -0.991390, "V14": -0.311169, "V15": 1.468177,
        "V16": -0.470401, "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
        "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928, "V25": 0.128539,
        "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
        "Amount": 149.62
    },
    "fraud": {
        "Time": 406.0,
        "V1": -2.312227, "V2": 1.951992, "V3": -1.609851, "V4": 3.997906, "V5": -0.522188,
        "V6": -1.426545, "V7": -2.537387, "V8": 1.391657, "V9": -2.770089, "V10": -2.772272,
        "V11": 3.202033, "V12": -2.899907, "V13": -0.595222, "V14": -4.289254, "V15": 0.389724,
        "V16": -1.140747, "V17": -2.830056, "V18": -0.016822, "V19": 0.416956, "V20": 0.126911,
        "V21": 0.517232, "V22": -0.035049, "V23": -0.465211, "V24": 0.320198, "V25": 0.044519,
        "V26": 0.177840, "V27": 0.261145, "V28": -0.143276,
        "Amount": 0.0
    }
}

# Load trained model
print("Loading model.pkl...")
model = pickle.load(open("model.pkl", "rb"))
print("Model loaded successfully.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/sample_data/<sample_type>", methods=["GET"])
def sample_data(sample_type):
    if sample_type in SAMPLE_TRANSACTIONS:
        return jsonify({"status": "success", "data": SAMPLE_TRANSACTIONS[sample_type]})
    return jsonify({"status": "error", "message": "Invalid sample type"}), 400

@app.route("/generate_features", methods=["GET"])
def generate_features():
    # Generate 28 random features using normal distribution
    v = np.random.normal(0, 1, 28)
    features = {f"V{i+1}": round(float(v[i]), 6) for i in range(28)}
    return jsonify({"status": "success", "data": features})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input JSON data provided"}), 400

        # Extract values in exact feature order
        row_values = []
        for col in FEATURE_NAMES:
            if col not in data:
                return jsonify({"error": f"Missing feature: {col}"}), 400
            row_values.append(float(data[col]))

        # Convert to DataFrame with matching column names to suppress feature warnings
        df_input = pd.DataFrame([row_values], columns=FEATURE_NAMES)

        # Predict class and probabilities
        prediction = int(model.predict(df_input)[0])
        probabilities = model.predict_proba(df_input)[0]
        
        fraud_prob = float(probabilities[1]) * 100.0
        safe_prob = float(probabilities[0]) * 100.0

        if fraud_prob >= 70.0:
            risk_level = "High Risk"
        elif fraud_prob >= 30.0:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Low Risk"

        is_fraud = bool(prediction == 1)
        result_str = "Fraud Detected ⚠️" if is_fraud else "Transaction Safe ✓"

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "is_fraud": is_fraud,
            "result": result_str,
            "fraud_probability": round(fraud_prob, 2),
            "safe_probability": round(safe_prob, 2),
            "risk_level": risk_level
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

