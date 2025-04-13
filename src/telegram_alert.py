import requests

def send_telegram_alert(bot_token, chat_id, message):
    """
    Sends an alert message via a Telegram bot.

    Parameters:
        bot_token (str): The HTTP API token for your Telegram bot (from BotFather).
        chat_id (str): The chat ID or channel ID where the message will be sent.
                       For a private chat, this can be your own user ID.
        message (str): The text to send.
    """
    url = f"https://api.telegram.org/bot{7849697100:AAEj_QP5i23TR7NJgfHZQ8O0EMxwth2vTVY}/sendMessage"
    payload = {
        "chat_id": 6438891757,
        "text": message
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Telegram alert sent successfully.")
        else:
            print(f"Failed to send Telegram alert: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error sending Telegram alert: {e}")
