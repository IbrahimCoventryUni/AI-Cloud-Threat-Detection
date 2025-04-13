import requests
from config import BULKSMS_CONFIG

def send_sms_alert(body, to_phone, from_phone):
    """
    Send an SMS alert using the BulkSMS API.
    
    Parameters:
        body (str): The SMS message content.
        to_phone (str): The recipient phone number in international format (as a string).
        from_phone (str): The sender ID or phone number.
    """
    headers = {
        "Authorization": BULKSMS_CONFIG.get("BASIC_AUTH"),
        "Content-Type": "application/json"
    }
    
    payload = {
        "to": to_phone,
        "body": body,
        "from": from_phone
    }
    
    api_url = BULKSMS_CONFIG.get("API_URL")
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code in (200, 201):
            print("SMS alert sent successfully.")
        else:
            print(f"Error sending SMS alert: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Exception sending SMS alert: {e}")
