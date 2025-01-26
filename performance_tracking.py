import json
import os
from datetime import datetime

LOG_PATH = "logs/performance_log.json"

def ensure_directories():
    if not os.path.exists("logs/"):
        os.makedirs("logs/")

def log_performance(model_version, precision, input_distribution):
    ensure_directories()

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": model_version,
        "precision": precision,
        "input_distribution": input_distribution,
    }

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r') as log_file:
            logs = json.load(log_file)
    else:
        logs = []

    logs.append(log_entry)

    with open(LOG_PATH, 'w') as log_file:
        json.dump(logs, log_file, indent=4)

    print(f"Performance logged: {log_entry}")
