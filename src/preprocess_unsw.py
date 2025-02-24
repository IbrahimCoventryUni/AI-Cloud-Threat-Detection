import pandas as pd  
import os  

# Define file paths  
data_dir = "../data/"  
csv_files = ["UNSW-NB15_1.csv", "UNSW-NB15_2.csv", "UNSW-NB15_3.csv", "UNSW-NB15_4.csv"]  

# Define column names  
columns = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl",
    "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime", "Sintpkt",
    "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd",
    "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label"
]

# Load and merge CSV files with proper column names  
dfs = [pd.read_csv(os.path.join(data_dir, file), names=columns, skiprows=1, low_memory=False) for file in csv_files]  
df = pd.concat(dfs, ignore_index=True)  

# Drop unnecessary columns (e.g., IP addresses for privacy)  
df.drop(columns=["srcip", "dstip"], inplace=True, errors='ignore')  

# Convert hexadecimal and categorical values to numerical  
for col in ["proto", "state", "service", "attack_cat"]:  
    df[col] = df[col].astype("category").cat.codes  # Convert category to numerical values  

# Convert hexadecimal values to integers (fixing '0x000c' issue)  
df = df.applymap(lambda x: int(x, 16) if isinstance(x, str) and x.startswith("0x") else x)  

# Ensure all data is numeric  
df = df.apply(pd.to_numeric, errors="coerce")  

# Save cleaned dataset  
df.to_csv("../data/cleaned_unsw.csv", index=False)  
print("✅ Preprocessing Complete. Cleaned data saved as 'cleaned_unsw.csv'.")  
