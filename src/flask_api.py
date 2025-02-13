from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load trained AI model
model_path = os.path.join(os.path.dirname(__file__), "../models/ai_threat_model.pkl")
model = joblib.load(model_path)

# Define API route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        # Convert result to human-readable format
        result = "BENIGN" if prediction[0] == "BENIGN" else "THREAT"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
