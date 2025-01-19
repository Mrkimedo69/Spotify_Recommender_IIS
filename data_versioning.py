import os
import pandas as pd
import json
from datetime import datetime

# Paths
DATA_DIR = "data_versions/"
LOG_DATA_PATH = "logs/data_version.log"

def ensure_data_directories():
    """
    Ensure that the data versions directory and logs exist.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists("logs/"):
        os.makedirs("logs/")

def save_dataset_version(df, version, description):
    """
    Save the dataset with a version and metadata.

    Args:
        df (pd.DataFrame): The dataset to save.
        version (str): The version of the dataset (e.g., "v1.0").
        description (str): Description of the dataset changes.
    """
    ensure_data_directories()
    
    # Save dataset as CSV
    dataset_path = os.path.join(DATA_DIR, f"data_{version}.csv")
    df.to_csv(dataset_path, index=False)
    
    # Save metadata as JSON
    metadata = {
        "version": version,
        "description": description,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_path": dataset_path
    }
    metadata_path = os.path.join(DATA_DIR, f"metadata_{version}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Dataset saved: {dataset_path}")
    print(f"Metadata saved: {metadata_path}")
    
    # Log the dataset version
    log_dataset_version(version, description, dataset_path)

def log_dataset_version(version, description, dataset_path):
    """
    Log dataset version details to a file.

    Args:
        version (str): Dataset version.
        description (str): Description of changes.
        dataset_path (str): Path to the dataset file.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] Version: {version}, Description: {description}, Path: {dataset_path}\n"
    
    with open(LOG_DATA_PATH, 'a') as log_file:
        log_file.write(log_entry)
    print(f"Dataset version logged: {log_entry}")

def load_dataset_version(version):
    """
    Load a specific dataset version and its metadata.

    Args:
        version (str): Version of the dataset to load.

    Returns:
        tuple: (pd.DataFrame, dict) - The dataset and its metadata.
    """
    dataset_path = os.path.join(DATA_DIR, f"data_{version}.csv")
    metadata_path = os.path.join(DATA_DIR, f"metadata_{version}.json")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found for version {version}: {dataset_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found for version {version}: {metadata_path}")
    
    df = pd.read_csv(dataset_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"Loaded dataset version {version} from {dataset_path}")
    return df, metadata

if __name__ == "__main__":
    ensure_data_directories()

    # Example dataset
    example_data = {
        "name": ["Song A", "Song B", "Song C"],
        "artists": [["Artist 1"], ["Artist 2"], ["Artist 3"]],
        "danceability": [0.8, 0.6, 0.7],
        "energy": [0.7, 0.5, 0.8],
        "tempo": [120, 130, 125],
        "valence": [0.5, 0.4, 0.6],
        "loudness": [-5, -6, -4],
    }
    df_example = pd.DataFrame(example_data)

    # Save dataset version 1.0
    save_dataset_version(df_example, version="v1.0", description="Initial version of the dataset.")

    # Simulate dataset changes
    df_example["popularity"] = [0.9, 0.7, 0.8]
    save_dataset_version(df_example, version="v1.1", description="Added popularity column.")

    # Load a specific version
    df_loaded, metadata_loaded = load_dataset_version("v1.1")
    print(metadata_loaded)
