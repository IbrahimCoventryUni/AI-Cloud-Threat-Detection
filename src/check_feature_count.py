import pandas as pd

# Load cleaned dataset
df = pd.read_csv("../data/cleaned_data.csv")

# Drop the label column if it exists
if 'Label' in df.columns:
    df = df.drop(columns=['Label'])

# Print the number of features
print(f"✅ Number of features expected by the model: {df.shape[1]}")
