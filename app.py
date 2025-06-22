import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the model
model = joblib.load("xgb_breast_cancer_model.pkl")

@app.route("/")
def loadPage():
    return render_template('home.html')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = data.get('features', None)

    if features is None or len(features) != 6:
        return jsonify({'error': 'Please provide 6 numeric features.'}), 400

    try:
        features = np.array(features, dtype=float).reshape(1, -1)
    except Exception:
        return jsonify({'error': 'Invalid feature values. Make sure all are numbers.'}), 400

    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]

    result = "Breast Cancer" if prediction == 1 else "Non-Breast Cancer"

    return jsonify({
        "prediction": result,
        "probability": round(float(proba), 4)
    })

# Ensure the server starts only when this file is run directly
if __name__ == "__main__":
    app.run(debug=True)