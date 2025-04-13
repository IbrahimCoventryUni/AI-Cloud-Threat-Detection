from flask import Flask, render_template_string
import os

app = Flask(__name__)
LOG_FILE = "threat_detection.log"

# Simple HTML template for the dashboard
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Threat Detection Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        pre { background-color: #f4f4f4; padding: 10px; border: 1px solid #ccc; }
    </style>
</head>
<body>
    <h1>Threat Detection Dashboard</h1>
    <p>Latest log entries:</p>
    <pre>{{ logs }}</pre>
</body>
</html>
"""

@app.route('/')
def dashboard():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            logs = f.read()
    else:
        logs = "No log file found."
    return render_template_string(HTML_TEMPLATE, logs=logs)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
    # browser : http://localhost:5001