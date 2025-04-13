from flask import Flask, jsonify
import os

app = Flask(__name__)
LOG_FILE = "threat_detection.log"

# In-memory status store (you can later populate it via your detection script)
status = {
    "last_alert_time": None,
    "total_alerts": 0,
    "system_status": "Running"
}

@app.route('/status', methods=['GET'])
def get_status():
    """
    Return current detection status and statistics.
    """
    return jsonify(status)

@app.route('/logs', methods=['GET'])
def get_logs():
    """
    Return the last N lines of the log file.
    """
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            # Read last 50 lines for example
            logs = f.readlines()[-50:]
        return jsonify({"logs": logs})
    else:
        return jsonify({"logs": "No log file found."}), 404

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5002)
#Go to http://localhost:5002/status to see current system status.

#Go to http://localhost:5002/logs to see recent log entries.