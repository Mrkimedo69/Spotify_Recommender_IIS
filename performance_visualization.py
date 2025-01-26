import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

LOG_PERFORMANCE_PATH = "logs/performance.log"

def parse_performance_log(log_path):

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    data = []
    with open(log_path, 'r') as file:
        for line in file:
            try:
                # Example log entry: [2025-01-18 22:53:25] Version: 1.1, Precision: 0.30, Description: Updated version of the model.
                timestamp = line.split("]")[0][1:]
                details = line.split("]")[1].strip()
                version = details.split(",")[0].split(":")[1].strip()
                precision = float(details.split(",")[1].split(":")[1].strip())
                description = details.split(",")[2].split(":")[1].strip()
                data.append({
                    "timestamp": datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S'),
                    "version": version,
                    "precision": precision,
                    "description": description
                })
            except Exception as e:
                print(f"Error parsing line: {line} | Error: {e}")
    return pd.DataFrame(data)


def plot_performance(df):
    df['version'] = df['version'].astype(str)
    
    plt.figure(figsize=(8, 6))
    plt.plot(df['version'], df['precision'], marker='o', label='Precision')
    plt.xlabel("Model Version")
    plt.ylabel("Precision")
    plt.title("Model Performance Over Versions")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.show()
    
if __name__ == "__main__":
    try:
        df_performance = parse_performance_log(LOG_PERFORMANCE_PATH)
        print(df_performance)

        # Plot the performance graph
        plot_performance(df_performance)
    except FileNotFoundError as e:
        print(e)
