# config.py
# Email configuration settings
EMAIL_CONFIG = {
    "EMAIL_SUBJECT": "Alert: Possible Cyber Attack Detected",
    "EMAIL_BODY": "A potential attack has been detected by the threat detection system. Please investigate immediately.",
    "TO_EMAIL": ["ibrahimaithreat@gmail.com", "mahaboobbi@coventry.ac.uk"],
    "FROM_EMAIL": "ibrahimaithreat@gmail.com",
    "SMTP_SERVER": "smtp.gmail.com",
    "SMTP_PORT": 587,
    "USERNAME": "ibrahimaithreat@gmail.com",
    "PASSWORD": "nstq bcif yier lvfr"  # Your app password
}

# Telegram configuration settings
TELEGRAM_CONFIG = {
    "BOT_TOKEN": "7849697100:AAEj_QP5i23TR7NJgfHZQ8O0EMxwth2vTVY",
    "CHAT_ID": "6438891757"
}

# Global alert cooldown period (in seconds)
ALERT_COOLDOWN = 60
