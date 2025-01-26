import os
import pandas as pd
import json
from datetime import datetime
from data_versioning import save_dataset_version

# Paths
DATA_PATH = "data.csv"
BASELINE_STATS_PATH = "baseline_stats.json"
LOG_DATA_PATH = "logs/data_version.log"

def calculate_baseline_stats(df):

    return {
        "columns": df.columns.tolist(),
        "num_rows": len(df),
        "stats": df.describe().to_dict()
    }

def detect_data_changes(baseline_stats, current_stats):

    changes = {}

    baseline_columns = set(baseline_stats.get("columns", []))
    current_columns = set(current_stats.get("columns", []))
    column_diff = current_columns.symmetric_difference(baseline_columns)
    if column_diff:
        changes["columns_changed"] = list(column_diff)

    for col in current_stats.get("stats", {}):
        if col in baseline_stats.get("stats", {}):
            baseline_col_stats = baseline_stats["stats"][col]
            current_col_stats = current_stats["stats"][col]

            for stat in current_col_stats:
                baseline_value = baseline_col_stats.get(stat, 0)
                current_value = current_col_stats[stat]
                if abs(current_value - baseline_value) > 0.1:
                    changes.setdefault(col, []).append(stat)

    return changes

def monitor_dataset():

    if not os.path.exists("logs/"):
        os.makedirs("logs/")

    df = pd.read_csv(DATA_PATH)

    current_stats = calculate_baseline_stats(df)

    if os.path.exists(BASELINE_STATS_PATH):
        with open(BASELINE_STATS_PATH, "r") as f:
            baseline_stats = json.load(f)

        changes = detect_data_changes(baseline_stats, current_stats)
        if changes:
            print("Significant changes detected in the dataset:")

            with open(LOG_DATA_PATH, "a") as log_file:
                log_file.write(f"\nDetected changes at {datetime.now()}:\n")
                log_file.write(json.dumps(changes, indent=4))
                log_file.write("\n" + "="*50 + "\n")

            version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"
            save_dataset_version(df, version=version, description="Dataset updated due to changes.")
            print(f"New dataset version saved: {version}")

            with open(BASELINE_STATS_PATH, "w") as f:
                json.dump(current_stats, f, indent=4)
            
            return changes  
        else:
            print("No significant changes detected in the dataset.")
            return {}
    else:

        with open(BASELINE_STATS_PATH, "w") as f:
            json.dump(current_stats, f, indent=4)
        print("Baseline statistics saved.")
        return {}

