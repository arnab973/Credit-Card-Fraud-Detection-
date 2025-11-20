from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the 7-feature model
model = pickle.load(open("model_7.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index_simple.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form - only the selected features
        features = ['V3', 'V7', 'V10', 'V12', 'V14', 'V16', 'V17', 'Time', 'Amount']
        input_data = []

        for feature in features:
            value = float(request.form[feature])
            input_data.append(value)

        # Make prediction
        prediction = model.predict([input_data])[0]

        if prediction == 0:
            output = "Valid Transaction ✔️"
        else:
            output = "Fraud Transaction ❌"

        return render_template('index_simple.html', output=output)

    except Exception as e:
        return render_template('index_simple.html', output=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
