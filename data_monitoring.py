import os
import pandas as pd
import json
from datetime import datetime
from data_versioning import save_dataset_version

# Paths
DATA_PATH = "data.csv"
BASELINE_STATS_PATH = "baseline_stats.json"
LOG_DATA_PATH = "logs/data_version.log"

# Function to calculate baseline statistics
def calculate_baseline_stats(df):
    """
    Calculate baseline statistics for the dataset.

    Args:
        df (pd.DataFrame): Dataset.

    Returns:
        dict: Baseline statistics.
    """
    return {
        "columns": df.columns.tolist(),
        "num_rows": len(df),
        "stats": df.describe().to_dict()
    }

# Function to detect changes in the dataset
def detect_data_changes(baseline_stats, current_stats):
    """
    Compare current dataset statistics with baseline statistics to detect changes.

    Args:
        baseline_stats (dict): Baseline statistics.
        current_stats (dict): Current dataset statistics.

    Returns:
        dict: Detected changes.
    """
    changes = {}

    # Check for column changes
    baseline_columns = set(baseline_stats.get("columns", []))
    current_columns = set(current_stats.get("columns", []))
    column_diff = current_columns.symmetric_difference(baseline_columns)
    if column_diff:
        changes["columns_changed"] = list(column_diff)

    # Check for statistical changes
    for col in current_stats.get("stats", {}):
        if col in baseline_stats.get("stats", {}):
            baseline_col_stats = baseline_stats["stats"][col]
            current_col_stats = current_stats["stats"][col]

            for stat in current_col_stats:
                baseline_value = baseline_col_stats.get(stat, 0)
                current_value = current_col_stats[stat]
                if abs(current_value - baseline_value) > 0.1:  # Tolerance
                    changes.setdefault(col, []).append(stat)

    return changes

# Main monitoring function
def main():
    # Ensure logs directory exists
    if not os.path.exists("logs/"):
        os.makedirs("logs/")

    # Load current dataset
    df = pd.read_csv(DATA_PATH)

    # Calculate current statistics
    current_stats = calculate_baseline_stats(df)

    # Check if baseline statistics exist
    if os.path.exists(BASELINE_STATS_PATH):
        with open(BASELINE_STATS_PATH, "r") as f:
            baseline_stats = json.load(f)

        # Detect changes
        changes = detect_data_changes(baseline_stats, current_stats)
        if changes:
            print("Significant changes detected in the dataset:")
            print(changes)

            # Save new dataset version
            version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"
            save_dataset_version(df, version=version, description="Dataset updated due to changes.")

            # Update baseline statistics
            with open(BASELINE_STATS_PATH, "w") as f:
                json.dump(current_stats, f, indent=4)
        else:
            print("No significant changes detected in the dataset.")
    else:
        # Save baseline statistics if not found
        with open(BASELINE_STATS_PATH, "w") as f:
            json.dump(current_stats, f, indent=4)
        print("Baseline statistics saved.")

if __name__ == "__main__":
    main()
