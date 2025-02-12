import pandas as pd
import numpy as np

# Load the cleaned dataset
file_path = "../data/cleaned_data.csv"
df = pd.read_csv(file_path)

# Strip spaces from column names
df.columns = df.columns.str.strip()

# Select only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Check for NaN values
nan_count = numeric_df.isna().sum().sum()

# Check for infinite values
inf_count = np.isinf(numeric_df).sum().sum()

# Check for extremely large values
max_value = numeric_df.max().max()

print(f"✅ NaN values found: {nan_count}")
print(f"✅ Infinite values found: {inf_count}")
print(f"✅ Maximum numeric value in dataset: {max_value}")
