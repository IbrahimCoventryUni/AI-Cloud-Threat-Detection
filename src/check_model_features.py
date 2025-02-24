import joblib
import os

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "../models/neural_network_unsw.pkl")
model = joblib.load(model_path)

# Check feature names
try:
    feature_names = model.feature_names_in_
    print("✅ Model expects these features:\n", feature_names)
except AttributeError:
    print("⚠️ Warning: Model does not store feature names.")
