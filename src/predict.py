import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load("../models/ai_threat_model.pkl")

# Define a function for real-time prediction
def predict_threat(sample_data):
    # Convert input data to DataFrame
    df_sample = pd.DataFrame([sample_data])

    # Ensure the order of columns matches training data
    trained_columns = pd.read_csv("../data/cleaned_data.csv").drop(columns=["Label"]).columns
    df_sample = df_sample.reindex(columns=trained_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(df_sample)
    return "🔵 BENIGN" if prediction[0] == "BENIGN" else "🚨 THREAT DETECTED"

# Example: Simulating real-time log entry
sample_entry = {
    " Destination Port": 80,
    " Flow Duration": 1500,
    " Total Fwd Packets": 10,
    " Total Backward Packets": 5,
    "Total Length of Fwd Packets": 500,
    " Total Length of Bwd Packets": 300,
    # Add remaining features with example values...
}

# Predict threat
result = predict_threat(sample_entry)
print(f"\n🔍 Prediction: {result}")
