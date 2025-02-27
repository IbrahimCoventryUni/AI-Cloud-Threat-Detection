import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

# Load the cleaned dataset
file_path = "../data/cleaned_unsw.csv"
df = pd.read_csv(file_path)

# Strip spaces from column names
df.columns = df.columns.str.strip()

# Check if 'attack_cat' and 'label' exist
if 'attack_cat' not in df.columns or 'label' not in df.columns:
    raise ValueError("❌ 'attack_cat' or 'label' column is missing! Please check dataset.")

# Separate features (X) and target variable (y)
X = df.drop(columns=['label', 'attack_cat'])  # Drop both 'label' and 'attack_cat'
y = df['attack_cat']  # Target variable

# Feature importance analysis using Extra Trees Classifier
model = ExtraTreesClassifier()
model.fit(X, y)

# Get feature importance scores
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance_sorted = feature_importance.sort_values(ascending=False)

# Print top 10 important features
print("✅ Top 10 Important Features:")
print(feature_importance_sorted[:10])

# Save feature importance scores
feature_importance_sorted.to_csv("../data/feature_importance.csv")
print("\n📂 'feature_importance.csv' has been saved in the data folder!")

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importance_sorted[:10].plot(kind='bar')
plt.title("Top 10 Important Features")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.xticks(rotation=45)
plt.savefig("../data/feature_importance.png")
plt.show()