from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("../models/random_forest_model.pkl")  # Ensure the correct path

# Define a threshold for attack classification
THRESHOLD = 0.3  # Adjust this based on testing
print("Model classes:", model.classes_)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])

        # Ensure column names match the trained model
        expected_features = [
            "sport", "dsport", "proto", "state", "dur", "sbytes", "dbytes", 
            "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload", 
            "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", 
            "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime", 
            "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat", 
            "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", 
            "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", 
            "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", 
            "syn_packet_rate", "is_syn_flood"
        ]

        df = df.reindex(columns=expected_features, fill_value=0)

        # 🚀 FORCE ATTACK DETECTION IF SYN RATE HIGH
        if df["syn_packet_rate"].iloc[0] > 10 or df["is_syn_flood"].iloc[0] == 1:
            prediction = "DDoS (Forced Detection)"
        else:
            prediction = model.predict(df)[0]

        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
