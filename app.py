from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate_features", methods=["GET"])
def generate_features():
    # Generate 28 random features using normal distribution
    v = np.random.normal(0, 1, 28)
    features = {f"V{i+1}": float(v[i]) for i in range(28)}
    return jsonify(features)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        # Extract values: Time, V1-V28, Amount
        values = [data["Time"]] + [data[f"V{i+1}"] for i in range(28)] + [data["Amount"]]
        values = np.array(values).reshape(1, -1)

        prediction = model.predict(values)[0]

        result = "Fraud Detected ⚠️" if prediction == 1 else "Transaction Safe ✓"

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
