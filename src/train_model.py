import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load preprocessed dataset
df_train = pd.read_csv('../data/cleaned_data_updated.csv')

# Check if 'label' column exists, rename if necessary
if "Label" in df_train.columns:
    df_train.rename(columns={"Label": "label"}, inplace=True)

# Ensure 'label' and 'attack_cat' columns are dropped correctly
df_train.drop(columns=["attack_cat"], errors="ignore", inplace=True)

# Encode the target variable
label_encoder = LabelEncoder()
df_train['label'] = label_encoder.fit_transform(df_train['label'])

# Define features and target
X = df_train.drop(columns=['label'])
y = df_train['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -------------------- Random Forest --------------------
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf_model = RandomForestClassifier(random_state=42)
rf_random = RandomizedSearchCV(rf_model, param_distributions=rf_params, n_iter=5, cv=3, verbose=2, n_jobs=-1, random_state=42)
rf_random.fit(X_train, y_train)

# Evaluate
rf_best = rf_random.best_estimator_
y_pred_rf = rf_best.predict(X_test)
print(f"🎯 Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# # Save model
# joblib.dump(rf_best, "../models/random_forest_cd.pkl")

# -------------------- XGBoost --------------------
xgb_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
xgb_random = RandomizedSearchCV(xgb_model, param_distributions=xgb_params, n_iter=5, cv=3, verbose=2, n_jobs=-1, random_state=42)
xgb_random.fit(X_train, y_train)

# Evaluate
xgb_best = xgb_random.best_estimator_
y_pred_xgb = xgb_best.predict(X_test)
print(f"🎯 XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(classification_report(y_test, y_pred_xgb, target_names=label_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

# # Save model
# joblib.dump(xgb_best, "../models/xgboost_cd.pkl")

# -------------------- Extra Trees --------------------
et_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
et_model = ExtraTreesClassifier(random_state=42)
et_random = RandomizedSearchCV(et_model, param_distributions=et_params, n_iter=5, cv=3, verbose=2, n_jobs=-1, random_state=42)
et_random.fit(X_train, y_train)

# Evaluate
et_best = et_random.best_estimator_
y_pred_et = et_best.predict(X_test)
print(f"🎯 Extra Trees Accuracy: {accuracy_score(y_test, y_pred_et):.4f}")
print(classification_report(y_test, y_pred_et, target_names=label_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_et))

# # Save model
# joblib.dump(et_best, "../models/extra_trees_cd.pkl")

print("✅ All models trained, optimized, and saved successfully!")