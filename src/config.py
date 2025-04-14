# config.py
# Email configuration settings
EMAIL_CONFIG = {
    "EMAIL_SUBJECT": "Alert: Possible Cyber Attack Detected",
    "EMAIL_BODY": "A potential attack has been detected by the threat detection system. Please investigate immediately.",
    "TO_EMAIL": ["my_email_xxx@xxx.com", "my_email_xxx@xxx.com"],
    "FROM_EMAIL": "my_email_xxx@xxx.com",
    "SMTP_SERVER": "smtp.gmail.com",
    "SMTP_PORT": 587,
    "USERNAME": "my_email_xxx@xxx.com",
    "PASSWORD": "my_actual_password123"
}

# Telegram configuration settings
TELEGRAM_CONFIG = {
    "BOT_TOKEN": "my_actual_token123",
    "CHAT_ID": "my_actual_token"
}

# Global alert cooldown period (in seconds)
ALERT_COOLDOWN = 60
