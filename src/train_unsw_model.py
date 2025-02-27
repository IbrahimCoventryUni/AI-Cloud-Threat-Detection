import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler

# 📌 Load the cleaned dataset
data_path = "../data/cleaned_unsw.csv"  # ✅ Ensure this is correct
df = pd.read_csv(data_path)

# 📌 Extract features and labels
X = df.drop(columns=["attack_cat"])  # Features
y = df["attack_cat"].replace(-1, 13)   # Labels

# 📌 Reduce the dataset size to avoid memory issues
sample_fraction = 0.3  # 🔥 Only use 30% of the dataset to fit in memory
X_sampled, _, y_sampled, _ = train_test_split(X, y, train_size=sample_fraction, random_state=42, stratify=y)

# 📌 Determine oversampling strategy (Limit max samples per class)
max_class_size = min(y_sampled.value_counts().max(), 100000)  # 🔥 Cap at 100K samples per class
sampling_strategy = {cls: max_class_size for cls, count in y_sampled.value_counts().items() if count < max_class_size}

# 📌 Apply oversampling (only increases minority class count)
oversampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X_sampled, y_sampled)

print(f"✅ Balanced dataset created: {len(X_resampled)} samples")

# 📌 Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# 📌 Normalize features using MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 📌 Save the scaler for later use in prediction
joblib.dump(scaler, "../models/feature_scaler.pkl")

# 📌 Compute class weights to handle imbalance in Neural Network
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}

print(df["attack_cat"].value_counts())


# 🚀 Train Random Forest Classifier
print("🚀 Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print(f"✅ Random Forest Accuracy: {rf_model.score(X_test, y_test):.4f}")
print(classification_report(y_test, rf_preds))

# 🚀 Train XGBoost Classifier
print("\n🚀 Training XGBoost...")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
print(f"✅ XGBoost Accuracy: {xgb_model.score(X_test, y_test):.4f}")
print(classification_report(y_test, xgb_preds))


# 🚀 Train Neural Network (MLP Classifier)
print("\n🚀 Training Neural Network...")
nn_model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                         max_iter=500, alpha=0.001, random_state=42)


# Replace NaN values with the column mean
X_train = np.nan_to_num(X_train, nan=np.nanmean(X_train))
X_test = np.nan_to_num(X_test, nan=np.nanmean(X_test))

# Verify if NaNs are removed
if np.isnan(X_train).sum() == 0 and np.isnan(X_test).sum() == 0:
    print("✅ No missing values in X_train and X_test")
else:
    print("⚠️ Warning: Missing values still exist")


nn_model.fit(X_train, y_train)
nn_preds = nn_model.predict(X_test)
print(f"✅ Neural Network Accuracy: {nn_model.score(X_test, y_test):.4f}")
print(classification_report(y_test, nn_preds))

# 📌 Save trained models
joblib.dump(rf_model, "../models/random_forest_unsw.pkl")
joblib.dump(xgb_model, "../models/xgboost_unsw.pkl")
joblib.dump(nn_model, "../models/neural_network_unsw.pkl")

print("\n✅ Model training completed and saved successfully!")
