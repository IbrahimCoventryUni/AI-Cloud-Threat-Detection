from scapy.all import sniff, IP, TCP, UDP
import requests
import time
import pandas as pd

# Flask API endpoint
API_URL = "http://127.0.0.1:5000/predict"

# Feature mapping (adjust based on your model's feature set)
def extract_features(packet):
    features = {
        "Destination Port": 5000,  # Assuming the attack is on port 5000
        "Flow Duration": 0,
        "Total Fwd Packets": 1,  # Single packet
        "Total Backward Packets": 0,  # Assuming no backward packets
        "Total Length of Fwd Packets": len(packet),
        "Total Length of Bwd Packets": 0,  # Assuming no backward packets
        "SYN Flag Count": int(packet[TCP].flags.S) if TCP in packet else 0,
        "ACK Flag Count": int(packet[TCP].flags.A) if TCP in packet else 0,
        "RST Flag Count": int(packet[TCP].flags.R) if TCP in packet else 0,
        # Add other features as needed
    }

    # Calculate Flow Duration (example: use current timestamp)
    features["Flow Duration"] = int(time.time())

    # Calculate Flow Bytes/s and Flow Packets/s
    features["Flow Bytes/s"] = features["Total Length of Fwd Packets"] / features["Flow Duration"]
    features["Flow Packets/s"] = features["Total Fwd Packets"] / features["Flow Duration"]

    return features

def process_packet(packet):
    """
    Process each packet: Extract features and send to Flask API for prediction.
    """
    try:
        # Filter packets from hping3 attack (destination port 5000)
        if (TCP in packet and packet[TCP].dport == 5000) or (UDP in packet and packet[UDP].dport == 5000):
            # Extract features from the packet
            features = extract_features(packet)

            # Debug: Print extracted features
            print(f"✅ Extracted features: {features}")

            # Send features to Flask API for prediction
            response = requests.post(
                API_URL,
                json={"features": features},
                headers={"Content-Type": "application/json"}
            )

            # Parse API response
            if response.status_code == 200:
                result = response.json()
                print(f"🔍 Model prediction probabilities: {result}")
                if result["prediction"] == "DDoS":
                    print(f"🚨 DDoS PACKET DETECTED! Details: {packet.summary()}")
                else:
                    print(f"✅ BENIGN PACKET: {packet.summary()}")
            else:
                print(f"⚠️ API Error: {response.text}")

    except Exception as e:
        print(f"⚠️ Error processing packet: {e}")

def start_sniffing():
    """
    Start sniffing packets and process them in real-time.
    """
    print("🚀 Starting packet capture...")
    sniff(iface="\\Device\\NPF_Loopback", prn=process_packet, filter="tcp or udp", store=False)

if __name__ == "__main__":
    start_sniffing()