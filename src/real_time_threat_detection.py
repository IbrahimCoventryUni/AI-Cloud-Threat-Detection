import os
import scapy.all as scapy
from scapy.layers.inet import IP, TCP, UDP
import joblib
import numpy as np

# Load the trained model from the models folder
model_path = os.path.join(os.path.dirname(__file__), "../models/random_forest_unsw.pkl")
model = joblib.load(model_path)

# Label mapping based on UNSW-NB15 dataset
attack_labels = {
    0: "BENIGN",
    1: "Generic",
    2: "Exploits",
    3: "Fuzzers",
    4: "DoS",
    5: "Reconnaissance",
    6: "Analysis",
    7: "Backdoor",
    8: "Shellcode",
    9: "Worms",
    10: "Shellcode",
    11: "Reconnaissance",
    12: "Analysis",
    13: "Fuzzers"
}

def extract_features(pkt):
    """
    Extracts features from a packet for ML model input.
    Returns a list of extracted features.
    """
    if pkt.haslayer(IP):
        sport = pkt.sport if pkt.haslayer(TCP) or pkt.haslayer(UDP) else 0
        dsport = pkt.dport if pkt.haslayer(TCP) or pkt.haslayer(UDP) else 0
        proto = pkt[IP].proto  # Protocol (TCP=6, UDP=17, etc.)
        sbytes = len(pkt)  # Packet size in bytes

        return [sport, dsport, proto, sbytes]
    else:
        return None  # Non-IP packets are ignored

def process_packet(pkt):
    """
    Processes a captured packet and classifies it using the trained ML model.
    """
    print(f"📜 Raw Packet Summary: {pkt.summary()}")  # Log packet details

    features = extract_features(pkt)
    if features is None:
        print("❌ Non-IP packet detected. Skipping...")
        return

    # Reshape for model input
    features_array = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features_array)[0]
    attack_class = attack_labels.get(prediction, "Unknown")

    print(f"✅ Packet {features[0]} → {features[1]} classified as: {attack_class}")

def start_sniffing():
    """
    Starts live packet sniffing.
    """
    print("🚀 Capturing network packets for real-time threat detection...")
    scapy.sniff(prn=process_packet, store=False)

if __name__ == "__main__":
    start_sniffing()