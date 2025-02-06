import pandas as pd

# Load the cleaned dataset
file_path = "../data/cleaned_data.csv"
df = pd.read_csv(file_path)

# Display basic information about the dataset
print("✅ Dataset Overview:")
print(df.info())

# Display the first few rows
print("\n✅ First 5 rows of the dataset:")
print(df.head())

# Check if there are any missing values
print("\n✅ Missing Values per Column:")
print(df.isnull().sum())

# Check the distribution of the target variable (attack types)
if 'Label' in df.columns:
    print("\n✅ Attack Type Distribution:")
    print(df['Label'].value_counts())

# Save a summary report
df.describe().to_csv("../data/data_summary.csv")
print("\n📂 'data_summary.csv' has been saved in the data folder!")
