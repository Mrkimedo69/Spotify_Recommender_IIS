import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from model_management import save_model

def train_model(train_df, features, k=5):
    print("Training model...")
    model = NearestNeighbors(n_neighbors=k, algorithm='auto')
    model.fit(train_df[features])
    print("Model training complete.")
    return model

def evaluate_model(test_df, model, features, tolerance=0.1):
    print("Evaluating model...")
    playlist_indices = np.random.choice(test_df.index, size=10, replace=False)
    playlist_features = test_df.loc[playlist_indices, features]

    distances, indices = model.kneighbors(playlist_features)

    precisions = []
    for i, idx_list in enumerate(indices):
        true_neighbors = test_df.iloc[idx_list]
        playlist_feature_mean = playlist_features.iloc[i].mean()
        within_tolerance = true_neighbors[features].apply(
            lambda x: np.all(np.abs(x - playlist_feature_mean) <= (tolerance * playlist_feature_mean)), axis=1)
        precisions.append(within_tolerance.mean())

    avg_precision = np.mean(precisions)
    print(f"Average Precision: {avg_precision:.2f}")
    return avg_precision

def training_pipeline(df, features, version, description, test_size=0.3, k=5, tolerance=0.1):
    print("Splitting dataset into training and testing sets...")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    print(f"Training set size: {len(train_df)}, Test set size: {len(test_df)}")
    
    model = train_model(train_df, features, k)
    precision = evaluate_model(test_df, model, features, tolerance)

    save_model(
        model, 
        version=version, 
        description=f"{description}. Precision: {precision:.2f}"
    )
    return model, precision

if __name__ == "__main__":
    file_path = 'data.csv'
    df = pd.read_csv(file_path)
    features = ['danceability', 'energy', 'tempo', 'valence', 'loudness']

    trained_model, precision = training_pipeline(
        df, 
        features, 
        version="1.0", 
        description="Initial training pipeline run"
    )
