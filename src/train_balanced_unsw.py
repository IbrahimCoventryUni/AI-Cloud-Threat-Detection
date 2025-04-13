import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("../data/balanced_unsw.csv")

# Check for missing values
print("🔍 Missing values before processing:\n", df.isnull().sum())

# Drop rows with missing values (OR fill them with mean/median)
df.dropna(inplace=True)  # Use df.fillna(df.mean(), inplace=True) if you want to fill NaNs instead
df.fillna(df.mean(), inplace=True)

# Separate features & labels
X = df.drop(columns=["attack_cat"])
y = df["attack_cat"]

# Convert -1 (benign) to class 0
y = y.replace(-1, 0)

# Apply SMOTE (oversampling minority classes)
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Save balanced dataset
balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
balanced_df["attack_cat"] = y_resampled
balanced_df.to_csv("../data/balanced_unsw_updated.csv", index=False)

print("✅ Balanced dataset saved as balanced_unsw_updated.csv")
print("🔍 Missing values after processing:\n", balanced_df.isnull().sum())
