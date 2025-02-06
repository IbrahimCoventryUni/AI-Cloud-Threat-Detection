import pandas as pd
import numpy as np

# Load the dataset
file_path = "../data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
df = pd.read_csv(file_path)

# Drop duplicate rows
df = df.drop_duplicates()

# Fill missing values with 0
df = df.fillna(0)

# Normalize IP addresses (convert to numeric format)
def ip_to_numeric(ip):
    parts = ip.split('.')
    return sum(int(part) * (256 ** i) for i, part in enumerate(reversed(parts)))

if 'Source IP' in df.columns:
    df['Source IP'] = df['Source IP'].apply(ip_to_numeric)
if 'Destination IP' in df.columns:
    df['Destination IP'] = df['Destination IP'].apply(ip_to_numeric)

# Normalize timestamp (convert to seconds)
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['Timestamp'] = (df['Timestamp'] - df['Timestamp'].min()).dt.total_seconds()

# Convert categorical values to machine-readable format
if 'Label' in df.columns:
    df['Label'] = df['Label'].astype('category').cat.codes

# Save the cleaned dataset
df.to_csv("../data/cleaned_data.csv", index=False)

print("✅ Data Preprocessing Completed. Cleaned data saved as 'cleaned_data.csv'")
