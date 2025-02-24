import requests
import time
import random

# Your deployed API URL
API_URL = "https://ai-cloud-threat-detection-production.up.railway.app/predict"
API_KEY = "9003121034"  

# Generate simulated feature values 
benign_sample = [54865.0, 3.0, 2.0, 0.0, 12.0, 0.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4000000.0, 666666.6667, 3.0, 0.0, 3.0, 3.0, 3.0, 3.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 40.0, 0.0, 666666.6667, 0.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 9.0, 6.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 12.0, 0.0, 0.0, 33.0, -1.0, 1.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Create a function to send requests
def simulate_traffic():
    while True:
        # Randomly decide to send a benign or threat request
        is_threat = random.choice([True, False])
        features = benign_sample.copy()

        if is_threat:
            # Modify some values to simulate a THREAT
            features[random.randint(0, len(features)-1)] += random.uniform(1000, 5000)
        
        payload = {"features": features}
        headers = {"x-api-key": API_KEY}

        try:
            response = requests.post(API_URL, json=payload, headers=headers)
            if response.status_code == 200:
                print(f"✅ {response.json()}")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ API Request Failed: {e}")

        # Wait a few seconds before sending the next request
        time.sleep(random.uniform(1, 5))

# Run the simulation
simulate_traffic()
