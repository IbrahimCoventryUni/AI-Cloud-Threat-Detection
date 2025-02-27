import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler

# Load dataset
data_path = "../data/cleaned_unsw.csv"
df = pd.read_csv(data_path)

# Load important features from feature_importance.csv
feature_importance_path = "../data/feature_importance.csv"
important_features = pd.read_csv(feature_importance_path, index_col=0).index.tolist()

# Keep only important features
selected_features = important_features[:20]  # Use only the top 20 features
X = df[selected_features]
y = df['attack_cat']

y = y.replace(-1, 13)

# Oversampling to balance dataset
oversampler = RandomOverSampler()
# Apply oversampling
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Reduce dataset size to 500,000 samples (adjust as needed)
sample_size = 500000
if len(X_resampled) > sample_size:
    X_resampled, y_resampled = X_resampled.sample(sample_size, random_state=42), y_resampled.sample(sample_size, random_state=42)

print(f"✅ Reduced dataset size: {len(X_resampled)} samples")

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=42)

# Train Random Forest
print("🚀 Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print(f"✅ Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf))

# Train XGBoost
print("\n🚀 Training XGBoost...")
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', tree_method='hist', max_depth=5)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print(f"✅ XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(classification_report(y_test, y_pred_xgb))

# # Train Neural Network
# print("\n🚀 Training Neural Network...")
# nn_model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=100)
# nn_model.fit(X_train, y_train)
# y_pred_nn = nn_model.predict(X_test)
# print(f"✅ Neural Network Accuracy: {accuracy_score(y_test, y_pred_nn):.4f}")
# print(classification_report(y_test, y_pred_nn))

# Save best model (Random Forest in this case)
joblib.dump(rf_model, "../models/random_forest_unsw.pkl")
print("✅ Model training completed and saved successfully!")
