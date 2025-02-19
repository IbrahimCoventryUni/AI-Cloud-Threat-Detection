import pandas as pd
import json

# Load cleaned data
df = pd.read_csv("../data/cleaned_data.csv")

# Drop the Label column
df = df.drop(columns=['Label'], errors='ignore')

# Take the first row as a sample input
sample_features = df.iloc[0].tolist()

# Convert to JSON format
sample_input = {"features": sample_features}

# Print JSON for cURL or Postman
print(json.dumps(sample_input, indent=4))
