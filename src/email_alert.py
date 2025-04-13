import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email_alert(subject, body, to_email, from_email, smtp_server, smtp_port, username, password):
    """
    Send an email alert using the specified SMTP server.
    
    Parameters:
        subject (str): Subject of the email.
        body (str): Body text of the email.
        to_email (list or str): Recipient email address(es). If a list is provided,
                                it will be joined into a comma-separated string.
        from_email (str): Sender email address.
        smtp_server (str): SMTP server address (e.g., 'smtp.gmail.com').
        smtp_port (int): SMTP server port (e.g., 587).
        username (str): Username for SMTP authentication.
        password (str): Password for SMTP authentication.
    """
    # Ensure to_email is a list; if not, make it a list
    if isinstance(to_email, str):
        to_email = [to_email]
    
    # Create a multipart message and set headers
    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = ', '.join(to_email)  # Join multiple addresses into one string
    msg["Subject"] = subject

    # Attach the email body
    msg.attach(MIMEText(body, "plain"))

    try:
        print("Connecting to SMTP server...")
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Secure the connection with TLS
        server.login(username, password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        print("Email alert sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")
