import os
import pickle
import json
from datetime import datetime

# Paths
MODEL_DIR = "model_store/"
LOG_PATH = "logs/model_version.log"
LOG_PERFORMANCE_PATH = os.path.abspath("logs/performance.log")

def log_performance(version, precision, description):
    if not os.path.exists("logs/"):
        os.makedirs("logs/")
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] Version: {version}, Precision: {precision:.2f}, Description: {description}\n"

    with open(LOG_PERFORMANCE_PATH, 'a') as log_file:
        log_file.write(log_entry)

    print(f"Performance logged: {log_entry}")

def ensure_directories():
    """
    Ensure that required directories for model storage exist.
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists("logs/"):
        os.makedirs("logs/")

# Function to save model with versioning
def save_model(model, version, description, dataset_version, directory="model_store"):
    ensure_directories()

    # Save the model
    model_path = os.path.join(directory, f"model_{version}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Save metadata
    metadata = {
        "version": version,
        "description": description,
        "dataset_version": dataset_version,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": model_path,
    }
    metadata_path = os.path.join(directory, f"metadata_{version}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Model saved: {model_path}")
    print(f"Metadata saved: {metadata_path}")

def log_model_version(version, description, model_path):
    """
    Log model version details to a file.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] Version: {version}, Description: {description}, Path: {model_path}\n"

    with open(LOG_PATH, 'a') as log_file:
        log_file.write(log_entry)

# Function to check if retraining is needed
def check_model_retraining(data_changes, threshold=1):
    """
    Provjerava trebaju li podaci pokrenuti ponovno treniranje modela.

    Args:
        data_changes (dict): Rječnik detektiranih promjena po stupcima i metrikama.
        threshold (int): Prag broja promjena za pokretanje retrainanja.

    Returns:
        bool: True ako model treba ponovno trenirati, False inače.
    """
    # Broji ukupan broj promjena u svim stupcima
    total_changes = sum(len(metrics) for metrics in data_changes.values())

    # Ako broj promjena prelazi prag, pokreće se ponovno treniranje
    return total_changes > threshold

# Function to load the latest model
def load_latest_model():
    """
    Load the latest model from the model store along with its metadata.

    Returns:
        tuple: Loaded model and its metadata.
    """
    ensure_directories()
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]

    if not models:
        print("No models found in the model store.")
        return None, None

    # Sort by timestamp in the filename
    models.sort(reverse=True)
    latest_model_name = models[0]
    latest_version = latest_model_name.split('_')[1].split('.')[0]
    latest_model_path = os.path.join(MODEL_DIR, latest_model_name)
    metadata_path = os.path.join(MODEL_DIR, f"metadata_{latest_version}.json")

    with open(latest_model_path, 'rb') as f:
        model = pickle.load(f)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"Loaded latest model: {latest_model_path}")
    return model, metadata

# Function to load a specific model version
def load_latest_model():
    """
    Load the latest model and its metadata from the model store.
    """
    ensure_directories()
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]

    if not models:
        print("No models found in the model store.")
        return None, None

    # Sort models by their modification time
    models.sort(key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)), reverse=True)

    for model_file in models:
        model_path = os.path.join(MODEL_DIR, model_file)
        metadata_file = f"metadata_{model_file.split('_')[1].replace('.pkl', '')}.json"
        metadata_path = os.path.join(MODEL_DIR, metadata_file)

        if not os.path.exists(metadata_path):
            print(f"Metadata file missing for model: {model_file}. Skipping...")
            continue

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print(f"Model loaded: {model_path}")
        print(f"Metadata loaded: {metadata_path}")
        return model, metadata

    print("No valid models with metadata found.")
    return None, None
def load_model(version):
    """
    Load a specific model and its metadata based on version.

    Args:
        version (str): The version of the model to load (e.g., "1.1").

    Returns:
        tuple: (model, metadata) - Loaded model and its metadata.
    """
    ensure_directories()
    
    # Define paths
    model_path = os.path.join(MODEL_DIR, f"model_{version}.pkl")
    metadata_path = os.path.join(MODEL_DIR, f"metadata_{version}.json")

    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found for version {version}: {model_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found for version {version}: {metadata_path}")

    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"Model loaded: {model_path}")
    print(f"Metadata loaded: {metadata_path}")
    return model, metadata

if __name__ == "__main__":
    ensure_directories()

    # Spremanje modela verzije 1.0
    dummy_model_v1 = {"example": "This is version 1.0 of the model."}
    save_model(dummy_model_v1, version="1.0", description="Initial version of the model.")

    # Spremanje modela verzije 1.1
    dummy_model_v1_1 = {"example": "This is version 1.1 of the model."}
    save_model(dummy_model_v1_1, version="1.1", description="Updated version of the model.")

    # Učitavanje specifične verzije modela
    model_v1_1, metadata_v1_1 = load_model("1.1")
    print(f"Loaded model version {metadata_v1_1['version']} metadata: {metadata_v1_1}")

