import time
import requests
import joblib
import pandas as pd
from scapy.all import sniff, IP, TCP, UDP, Raw
import numpy as np
import logging

from config import EMAIL_CONFIG, TELEGRAM_CONFIG, ALERT_COOLDOWN

######################################
# 1. LOGGER SETUP
######################################
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ThreatDetection")

######################################
# 2. ALERT FUNCTIONS
######################################
def send_email_alert(subject, body, to_email, from_email, smtp_server, smtp_port, username, password):
    """
    Sends an email alert using smtplib.
    """
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    msg = MIMEMultipart()
    msg["From"] = from_email
    if isinstance(to_email, str):
        to_email = [to_email]
    msg["To"] = ", ".join(to_email)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(username, password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        logger.info("Email alert sent successfully.")
    except Exception as e:
        logger.error(f"Error sending email: {e}")

def send_telegram_alert(bot_token, chat_id, message):
    """
    Sends a Telegram message using the Bot API.
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            logger.info("Telegram alert sent successfully.")
        else:
            logger.error(f"Failed to send Telegram alert: {response.status_code}, {response.text}")
    except Exception as e:
        logger.error(f"Error sending Telegram alert: {e}")

######################################
# 3. CONFIGURATION FROM config.py
######################################
EMAIL_SUBJECT = EMAIL_CONFIG.get("EMAIL_SUBJECT")
EMAIL_BODY = EMAIL_CONFIG.get("EMAIL_BODY")
TO_EMAIL = EMAIL_CONFIG.get("TO_EMAIL")
FROM_EMAIL = EMAIL_CONFIG.get("FROM_EMAIL")
SMTP_SERVER = EMAIL_CONFIG.get("SMTP_SERVER")
SMTP_PORT = EMAIL_CONFIG.get("SMTP_PORT")
USERNAME = EMAIL_CONFIG.get("USERNAME")
PASSWORD = EMAIL_CONFIG.get("PASSWORD")

TELEGRAM_BOT_TOKEN = TELEGRAM_CONFIG.get("BOT_TOKEN")
TELEGRAM_CHAT_ID = TELEGRAM_CONFIG.get("CHAT_ID")

######################################
# 4. GLOBAL ALERT STATE
######################################
# Define the global alert throttle variable at the top so it's available in all functions.
last_alert_sent = 0

######################################
# 5. LOAD MODELS AND SCALER
######################################
try:
    rf_model = joblib.load("../models/random_forest_model.pkl")
    xgb_model = joblib.load("../models/xgboost_model.pkl")
    et_model = joblib.load("../models/extra_trees_model.pkl")
    scaler = joblib.load("../models/feature_scaler.pkl")

    model_features = rf_model.feature_names_in_
    scaler_features = scaler.feature_names_in_
    assert set(scaler_features) <= set(model_features), "Scaler and model feature mismatch!"
except Exception as e:
    logger.error(f"Error loading models or scaler: {e}")
    exit(1)

######################################
# 6. GLOBAL DICTIONARIES FOR TRACKING PACKETS
######################################
syn_packet_counts = {}
udp_packet_counts = {}
http_packet_counts = {}

######################################
# 7. HELPER FUNCTIONS
######################################
def update_udp_count(packet):
    """Track UDP packet count over the last 5 seconds."""
    if packet.haslayer(UDP):
        current_time = int(time.time())
        if current_time not in udp_packet_counts:
            udp_packet_counts[current_time] = 0
        udp_packet_counts[current_time] += 1
        last_5_seconds = [udp_packet_counts.get(current_time - i, 0) for i in range(5)]
        return sum(last_5_seconds)
    return 0

def update_http_count(packet):
    """Track HTTP GET request count over the last 5 seconds."""
    if packet.haslayer(TCP) and (packet[TCP].dport in [80, 443]):
        if packet.haslayer(Raw):
            payload = packet.getlayer(Raw).load
            if isinstance(payload, bytes) and payload.startswith(b"GET"):
                current_time = int(time.time())
                if current_time not in http_packet_counts:
                    http_packet_counts[current_time] = 0
                http_packet_counts[current_time] += 1
                last_5_seconds = [http_packet_counts.get(current_time - i, 0) for i in range(5)]
                return sum(last_5_seconds)
    return 0

def extract_features(packet):
    """Extract features from a packet for forced and model-based detection."""
    global syn_packet_counts

    sport = packet.sport if hasattr(packet, 'sport') else 0
    dsport = packet.dport if hasattr(packet, 'dport') else 0
    proto = packet.proto if hasattr(packet, 'proto') else 0

    # Check SYN flag (0x02)
    state = 0
    if packet.haslayer(TCP) and (packet[TCP].flags & 2):
        state = 1

    # Track SYN rate over the last 5 seconds
    current_time = int(time.time())
    if current_time not in syn_packet_counts:
        syn_packet_counts[current_time] = 0
    if state == 1:
        syn_packet_counts[current_time] += 1
    last_5_seconds_syn = [syn_packet_counts.get(current_time - i, 0) for i in range(5)]
    syn_packet_rate = sum(last_5_seconds_syn)
    is_syn_flood = 1 if syn_packet_rate > 20 else 0

    # Minimal placeholders for additional features
    dur = sbytes = dbytes = sttl = dttl = sloss = dloss = service = Sload = Dload = 0
    Spkts = Dpkts = swin = dwin = stcpb = dtcpb = smeansz = dmeansz = trans_depth = 0
    res_bdy_len = Sjit = Djit = Sintpkt = Dintpkt = tcprtt = synack = ackdat = 0
    is_sm_ips_ports = ct_state_ttl = ct_flw_http_mthd = is_ftp_login = ct_ftp_cmd = 0
    ct_srv_src = ct_srv_dst = ct_dst_ltm = ct_src_ltm = ct_src_dport_ltm = ct_dst_sport_ltm = ct_dst_src_ltm = 0
    Stime = current_time
    Ltime = current_time + 1

    feature_values = [
        sport, dsport, proto, state, dur, sbytes, dbytes, sttl, dttl, sloss, dloss, service,
        Sload, Dload, Spkts, Dpkts, swin, dwin, stcpb, dtcpb, smeansz, dmeansz, trans_depth,
        res_bdy_len, Sjit, Djit, Stime, Ltime, Sintpkt, Dintpkt, tcprtt, synack, ackdat,
        is_sm_ips_ports, ct_state_ttl, ct_flw_http_mthd, is_ftp_login, ct_ftp_cmd, ct_srv_src,
        ct_srv_dst, ct_dst_ltm, ct_src_ltm, ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm,
        syn_packet_rate, is_syn_flood
    ]
    feature_names = [
        'sport', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl',
        'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin',
        'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit',
        'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports',
        'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src',
        'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
        'ct_dst_src_ltm', 'syn_packet_rate', 'is_syn_flood'
    ]
    return pd.DataFrame([feature_values], columns=feature_names)

######################################
# 8. PACKET PROCESSING (UNIFIED FORCED CHECK)
######################################
def process_packet(packet):
    global last_alert_sent
    try:
        udp_rate = update_udp_count(packet)
        http_rate = update_http_count(packet)
        feature_df = extract_features(packet)

        logger.info("Extracted Features:\n%s", feature_df)

        current_time = time.time()
        forced_attack_type = None

        # Unified forced detection block
        syn_rate = feature_df["syn_packet_rate"].iloc[0]
        is_syn_flood = feature_df["is_syn_flood"].iloc[0]

        if syn_rate > 10 or is_syn_flood == 1:
            forced_attack_type = f"SYN Flood (rate={syn_rate})"
        elif udp_rate > 10:
            forced_attack_type = f"UDP Flood (rate={udp_rate})"
        elif http_rate > 10:
            forced_attack_type = f"HTTP Flood (rate={http_rate})"

        if forced_attack_type:
            logger.warning(f"Threat detected! Forced DDoS Detection triggered - {forced_attack_type}")
            time_since_last = current_time - last_alert_sent
            logger.info(f"Time since last alert: {time_since_last:.2f} seconds")

            if time_since_last > ALERT_COOLDOWN:
                send_email_alert(
                    EMAIL_SUBJECT,
                    f"A forced attack was detected: {forced_attack_type}\nPlease investigate.",
                    TO_EMAIL,
                    FROM_EMAIL,
                    SMTP_SERVER,
                    SMTP_PORT,
                    USERNAME,
                    PASSWORD
                )
                send_telegram_alert(
                    TELEGRAM_BOT_TOKEN,
                    TELEGRAM_CHAT_ID,
                    f"Alert! A forced attack was detected: {forced_attack_type}"
                )
                last_alert_sent = current_time
            return

        # Model-based detection if no forced detection triggers
        feature_scaled = scaler.transform(feature_df[scaler_features])
        feature_scaled_df = pd.DataFrame(feature_scaled, columns=scaler_features)

        rf_pred = rf_model.predict(feature_scaled_df)
        xgb_pred = xgb_model.predict(feature_scaled_df)
        et_pred = et_model.predict(feature_scaled_df)

        final_prediction = np.round((rf_pred + xgb_pred + et_pred) / 3).astype(int)
        logger.info("Model Raw Prediction: %s", final_prediction)

        if final_prediction[0] == 1:
            logger.warning("Threat detected! Possible attack packet captured (Model-based).")
        else:
            logger.info("Normal traffic detected (Model-based).")

    except Exception as e:
        logger.error(f"Error processing packet: {e}")

######################################
# 9. START PACKET CAPTURE
######################################
def capture_packets():
    logger.info("Monitoring network traffic...")
    sniff(
        iface="\\Device\\NPF_Loopback",
        prn=process_packet,
        filter="tcp or udp and (port 80 or port 443 or port 53 or port 22 or port 5000)",
        store=False
    )

if __name__ == "__main__":
    capture_packets()
