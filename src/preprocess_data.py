import pandas as pd
import numpy as np

# Load the dataset
file_path = "../data/cleaned_data.csv"
df = pd.read_csv(file_path)

# Strip spaces from column names
df.columns = df.columns.str.strip()

# Replace infinite values with the maximum finite value of each column
numeric_cols = df.select_dtypes(include=[np.number])
for col in numeric_cols.columns:
    max_finite = numeric_cols[col].replace([np.inf, -np.inf], np.nan).max()
    df[col] = df[col].replace([np.inf, -np.inf], max_finite)

# Save the fixed dataset
df.to_csv("../data/cleaned_data.csv", index=False)
print("✅ Data Cleaning Completed. Infinite values replaced.")
