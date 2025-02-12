import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load cleaned data
df = pd.read_csv("../data/cleaned_data.csv")

# Separate features and labels
X = df.drop(columns=["Label"])  # Features
y = df["Label"]  # Target variable

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save trained model
joblib.dump(model, "../models/ai_threat_model.pkl")

print(f"✅ Model training complete! Accuracy: {accuracy:.4f}")
print("✅ Model saved as 'ai_threat_model.pkl'")
