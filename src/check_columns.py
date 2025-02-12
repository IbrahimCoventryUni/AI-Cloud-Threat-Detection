import pandas as pd

# Load the cleaned dataset
file_path = "../data/cleaned_data.csv"
df = pd.read_csv(file_path)

# Print all column names
print("✅ Column Names in cleaned_data.csv:")
print(df.columns)
