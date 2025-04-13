import requests
import time
import random
import json

# Your deployed API URL
API_URL = "http://127.0.0.1:5000/predict"
API_KEY = "9003121034"  

# Simulated network traffic data (EXACTLY 46 values)
features = [
    54865.0, 3.0, 2.0, 0.0, 12.0, 0.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    4000000.0, 666666.6667, 3.0, 0.0, 3.0, 3.0
]

# Prepare request payload
data = {"features": features}

# Set request headers
headers = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY  # Ensure correct API key
}

try:
    # Send POST request to API
    response = requests.post(API_URL, headers=headers, json=data)

    # Print response
    print("Response Status Code:", response.status_code)
    print("Response JSON:", response.json())

    # Handle different responses
    if response.status_code == 200:
        print("✅ Threat detection successful!")
    elif response.status_code == 403:
        print("🔴 Error: Unauthorized. Check API key.")
    elif response.status_code == 400:
        print("🔴 Error: Bad request. Check JSON format & feature count.")
    else:
        print("⚠️ Unexpected response:", response.text)

except requests.exceptions.RequestException as e:
    print("🔴 Request failed:", str(e))