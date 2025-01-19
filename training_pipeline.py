import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
from model_management import save_model

# Funkcija za treniranje modela
def train_model(df, features, k=5):
    """
    Train a nearest neighbors model on the given data.
    """
    print("Training model...")
    model = NearestNeighbors(n_neighbors=k, algorithm='auto')
    model.fit(df[features])
    print("Model training complete.")
    return model

# Funkcija za evaluaciju modela
def evaluate_model(df, model, features, tolerance=0.1):
    """
    Evaluate the trained model using a simple precision metric.
    """
    print("Evaluating model...")
    playlist_indices = np.random.choice(df.index, size=10, replace=False)  # Random playlist for evaluation
    playlist_features = df.iloc[playlist_indices][features]

    # Calculate neighbors
    distances, indices = model.kneighbors(playlist_features)

    # Precision as a percentage of neighbors within tolerance
    precisions = []
    for i, idx_list in enumerate(indices):
        true_neighbors = df.iloc[idx_list]
        playlist_feature_mean = playlist_features.iloc[i].mean()
        within_tolerance = true_neighbors[features].apply(
            lambda x: np.all(np.abs(x - playlist_feature_mean) <= (tolerance * playlist_feature_mean)), axis=1)
        precisions.append(within_tolerance.mean())

    avg_precision = np.mean(precisions)
    print(f"Average Precision: {avg_precision:.2f}")
    return avg_precision

# Automatski pipeline za treniranje i evaluaciju
def training_pipeline(df, features, version, description, k=5, tolerance=0.1):
    """
    Full pipeline to train, evaluate, and save a model.
    """
    model = train_model(df, features, k)
    precision = evaluate_model(df, model, features, tolerance)

    # Save model with evaluation details
    save_model(
        model, 
        version=version, 
        description=f"{description}. Precision: {precision:.2f}"
    )
    return model, precision

if __name__ == "__main__":
    # Load and preprocess dataset
    file_path = 'data.csv'
    df = pd.read_csv(file_path)
    features = ['danceability', 'energy', 'tempo', 'valence', 'loudness']

    # Example usage
    trained_model, precision = training_pipeline(
        df, 
        features, 
        version="1.0", 
        description="Initial training pipeline run"
    )
