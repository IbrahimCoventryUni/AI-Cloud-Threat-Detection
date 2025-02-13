import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# Load cleaned data
df = pd.read_csv("../data/cleaned_data.csv")

# Separate features and labels
X = df.drop(columns=["Label"])
y = df["Label"]

# Load trained model
model = joblib.load("../models/ai_threat_model.pkl")

# Make predictions
y_pred = model.predict(X)

# Generate evaluation metrics
print("\n✅ Model Evaluation:")
print(confusion_matrix(y, y_pred))
print("\nClassification Report:")
print(classification_report(y, y_pred))
