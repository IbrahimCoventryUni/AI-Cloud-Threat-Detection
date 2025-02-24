import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("../data/cleaned_unsw.csv")

# Separate attack and benign samples
benign_df = df[df["label"] == 0]
attack_df = df[df["label"] == 1]

# Balance by undersampling majority class
benign_sample = benign_df.sample(n=len(attack_df), random_state=42)

# Merge balanced dataset
df_balanced = pd.concat([benign_sample, attack_df])

# Shuffle dataset
df_balanced = df_balanced.sample(frac=1, random_state=42)

# Separate features and labels
X = df_balanced.drop(columns=["label"])
y = df_balanced["label"]


# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Fix: Ensure DataFrame is copied to avoid modification warnings
X_train = X_train.copy()
X_test = X_test.copy()

# ✅ Fix: Fill missing values for `sport` and `dsport` with most frequent value (mode)
for col in ["sport", "dsport"]:
    if col in X_train.columns:
        X_train[col] = X_train[col].fillna(X_train[col].mode()[0])
        X_test[col] = X_test[col].fillna(X_test[col].mode()[0])

# ✅ Fix: Handle `ct_ftp_cmd` only if it exists
if "ct_ftp_cmd" in X_train.columns:
    X_train["ct_ftp_cmd"] = X_train["ct_ftp_cmd"].fillna(0)
if "ct_ftp_cmd" in X_test.columns:
    X_test["ct_ftp_cmd"] = X_test["ct_ftp_cmd"].fillna(0)

# 🚀 Train Random Forest with stronger regularization
print("🚀 Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=30,      # Fewer trees
    max_depth=6,          # Even shallower depth
    min_samples_split=30, # Larger split requirement
    random_state=42
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print(f"✅ Random Forest Accuracy: {rf_model.score(X_test, y_test):.4f}")
print(classification_report(y_test, y_pred_rf))

# Save model
joblib.dump(rf_model, "../models/random_forest_unsw.pkl")

# 🚀 Train XGBoost with stronger regularization
print("\n🚀 Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=30,      
    max_depth=6,          
    learning_rate=0.05,   # Lower learning rate  
    reg_lambda=2.0,       # More L2 regularization  
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print(f"✅ XGBoost Accuracy: {xgb_model.score(X_test, y_test):.4f}")
print(classification_report(y_test, y_pred_xgb))

# Save model
joblib.dump(xgb_model, "../models/xgboost_unsw.pkl")

# 🚀 Train Neural Network (MLP)
print("\n🚀 Training Neural Network...")
mlp_model = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=1000, random_state=42, early_stopping=True, learning_rate="adaptive")
mlp_model.fit(X_train, y_train)
y_pred_mlp = mlp_model.predict(X_test)
print(f"✅ Neural Network Accuracy: {mlp_model.score(X_test, y_test):.4f}")
print(classification_report(y_test, y_pred_mlp, zero_division=1))

# Save model
joblib.dump(mlp_model, "../models/neural_network_unsw.pkl")

print("\n✅ Model training completed and saved successfully!")
