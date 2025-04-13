import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the cleaned dataset used for training
df = pd.read_csv("../data/balanced_unsw_updated2.csv")

# Select only the features used by the model
FEATURES = [
    "sport", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl",
    "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin",
    "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit",
    "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "syn_packet_rate",
    "is_syn_flood"
]

df = df[FEATURES]  # Keep only relevant features

# Train a new MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df)

# Save the updated scaler
joblib.dump(scaler, "../models/feature_scaler.pkl")
print("✅ MinMaxScaler retrained and saved successfully!")
