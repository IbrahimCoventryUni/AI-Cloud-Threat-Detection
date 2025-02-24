import pandas as pd  
import numpy as np  
import joblib  
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier  
from sklearn.neural_network import MLPClassifier  
from sklearn.metrics import accuracy_score, classification_report  

# Load preprocessed dataset  
df = pd.read_csv("../data/cleaned_unsw.csv")  


# Separate features and labels  
X = df.drop(columns=["label"])  
y = df["label"]  

# Split dataset (80% train, 20% test)  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Define models  
models = {  
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),  
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),  
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)  
}  

# Train & Evaluate Each Model  
for name, model in models.items():  
    print(f"🚀 Training {name}...")  
    model.fit(X_train, y_train)  
    y_pred = model.predict(X_test)  
    print(f"✅ {name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")  
    print(classification_report(y_test, y_pred))  

    # Save the trained model  
    joblib.dump(model, f"../models/{name.replace(' ', '_').lower()}_unsw.pkl")  

print("🎯 Model training completed. Models saved in 'models/' folder.")  
