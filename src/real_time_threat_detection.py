import joblib
import requests
import scapy.all as scapy
import pandas as pd

# Load the trained model
model_path = "../models/random_forest_unsw.pkl"
model = joblib.load(model_path)

# Load the selected features from feature_importance.csv
feature_importance_path = "../data/feature_importance.csv"
important_features = pd.read_csv(feature_importance_path, index_col=0).index.tolist()
selected_features = important_features[:20]  # Match features used in training

# Flask API URL
API_URL = "http://127.0.0.1:5000/predict"

def extract_features(pkt):
    """Extract relevant features from a network packet."""
    if not pkt.haslayer(scapy.IP):
        print("❌ Non-IP packet detected. Skipping...")
        return None

    # Extract basic packet features
    sport = pkt.sport if hasattr(pkt, 'sport') else 0
    dsport = pkt.dport if hasattr(pkt, 'dport') else 0
    proto = pkt.proto if hasattr(pkt, 'proto') else -1
    sbytes = len(pkt) if hasattr(pkt, 'len') else 0

    # Construct feature dictionary
    features_dict = {
        "sport": sport,
        "dsport": dsport,
        "proto": proto,
        "sbytes": sbytes
    }

    # Ensure all selected features exist in dictionary (fill missing with 0)
    feature_vector = [features_dict.get(f, 0) for f in selected_features]
    
    return feature_vector

def detect_threat(pkt):
    """Process a packet and send features to the Flask API."""
    print(f"📜 Raw Packet Summary: {pkt.summary()}")

    features = extract_features(pkt)
    if features is None:
        return
    
    data = {"features": features}
    response = requests.post(API_URL, json=data)
    
    if response.status_code == 200:
        result = response.json()["prediction"]
        print(f"✅ Packet {features[0]} → {features[1]} classified as: {result}")
    else:
        print("🔴 Error processing packet:", response.text)

# Start sniffing network traffic
print("🚀 Capturing network packets for real-time threat detection...")
scapy.sniff(prn=detect_threat, store=False)
