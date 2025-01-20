import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score
import gradio as gr
from performance_visualization import parse_performance_log, plot_performance
from training_pipeline import training_pipeline
from model_management import check_model_retraining, log_performance
from data_versioning import load_dataset_version

# Function: Data Cleaning and Preprocessing
def clean_and_preprocess_data(file_path=None, df=None):
    if df is None and file_path is None:
        raise ValueError("Either 'file_path' or 'df' must be provided.")
    if df is None:
        df = pd.read_csv(file_path)

    # Remove columns and rows with excessive missing values
    missing_row_threshold = 0.5
    missing_col_threshold = 0.5
    df = df.loc[:, df.isnull().mean() < missing_col_threshold]
    df = df[df.isnull().mean(axis=1) < missing_row_threshold]

    # Fill missing values for numerical columns
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].mean())

    # Fill missing values for categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('Unknown')

    # Drop duplicate rows based on song name
    if 'name' in df.columns:
        df = df.drop_duplicates(subset=['name'], keep='first').reset_index(drop=True)

    # Normalization and transformations
    min_max_features = ['tempo', 'valence', 'popularity']
    z_score_features = ['danceability', 'energy']
    log_transform_features = ['loudness']

    # Min-Max Scaling (check if columns exist)
    existing_min_max_features = [col for col in min_max_features if col in df.columns]
    if existing_min_max_features:
        scaler_min_max = MinMaxScaler()
        df[existing_min_max_features] = scaler_min_max.fit_transform(df[existing_min_max_features])

    # Z-score Standardization (check if columns exist)
    existing_z_score_features = [col for col in z_score_features if col in df.columns]
    if existing_z_score_features:
        scaler_z_score = StandardScaler()
        df[existing_z_score_features] = scaler_z_score.fit_transform(df[existing_z_score_features])

    # Log Transformation (check if columns exist)
    for feature in log_transform_features:
        if feature in df.columns:
            df[feature] = np.log1p(df[feature] - df[feature].min() + 1)

    return df
# Function: Evaluate Precision
def evaluate_similarity_precision(df, playlist_indices, recommended_indices, features, tolerance=0.1):
    if not playlist_indices or not recommended_indices:
        return 0.0

    # Get average features of the playlist
    playlist_features = df.iloc[playlist_indices][features].mean().values

    # Get features of recommended songs
    recommended_features = df.iloc[recommended_indices][features].values

    # Calculate similarity
    similarities = cosine_similarity([playlist_features], recommended_features).flatten()

    # Determine relevant songs based on tolerance
    relevant = (similarities >= (1 - tolerance)).astype(int)
    predicted = [1] * len(recommended_indices)  # All recommended songs are predictions

    # Calculate precision
    return precision_score(relevant, predicted)

# Function: Generate Recommendations
def generate_recommendations(df, search_artist, search_song, features, top_n=10):
    playlist_indices = []
    if search_artist:
        playlist_indices = df[df['artists'].str.contains(search_artist, case=False, na=False)].index.tolist()
    elif search_song:
        playlist_indices = df[df['name'].str.contains(search_song, case=False, na=False)].index.tolist()

    if not playlist_indices:
        return pd.DataFrame(), f"Sorry, no results found for '{search_artist or search_song}'.", [], []

    # Prosječne značajke za playlistu
    playlist_features = df.iloc[playlist_indices][features]
    average_playlist_features = playlist_features.mean(axis=0).values.reshape(1, -1)

    # Kopiraj DataFrame za sigurnu obradu
    df_copy = df.copy()

    # Računaj sličnost i filtriraj preporuke
    df_copy['similarity'] = cosine_similarity(average_playlist_features, df_copy[features]).flatten()

    # Ukloni pjesme iz trenutne playliste
    df_copy = df_copy[~df_copy.index.isin(playlist_indices)]

    # Sortiraj prema sličnosti
    df_copy = df_copy.sort_values(by='similarity', ascending=False)

    # Odaberi top_n preporuka
    recommended_songs = df_copy.head(top_n)

    # Generiraj objašnjenja
    explanations = [f"This song is recommended because of its similarity in {', '.join(features)}." for _ in range(len(recommended_songs))]
    return recommended_songs, "", playlist_indices, explanations

# GRadio Interface
def gradio_interface(search_artist, search_song, tolerance, current_size, increment):
    LOG_PERFORMANCE_PATH = os.path.abspath("logs/performance.log")
    global current_df, df

    if not search_artist and not search_song:
        return "<b>Please enter either an artist's name or a song name.</b>", ""

    # Adjust dataset size dynamically
    if current_size > len(df):
        current_size = len(df)
    current_df = df.head(int(current_size))

    # Generate recommendations
    recommended_songs, rec_error, playlist_indices, explanations = generate_recommendations(
        current_df, search_artist, search_song, features, top_n=10
    )

    if rec_error:
        return f"<b>{rec_error}</b>", ""

    # Calculate Precision
    precision = evaluate_similarity_precision(
        df=current_df,
        playlist_indices=playlist_indices,
        recommended_indices=recommended_songs.index.tolist(),
        features=features,
        tolerance=tolerance,
    )
    log_performance(version="1.1", precision=precision, description="Updated version of the model.")

    try:
        df_performance = parse_performance_log(LOG_PERFORMANCE_PATH)
        plot_performance(df_performance)
    except Exception as e:
        print(f"Error generating performance visualization: {e}")

    # Format recommendations with explanations as tooltips
    recommended_songs['Explanation'] = explanations
    recommended_songs_output = recommended_songs[['name', 'artists', 'popularity', 'Explanation']].copy()
    recommended_songs_output['artists'] = recommended_songs_output['artists'].apply(lambda x: ', '.join(eval(x)))
    recommended_songs_output_html = recommended_songs_output.to_html(index=False, escape=False, classes='table table-striped')

    # Add legends dynamically
    if search_artist:
        legend = """<table class='table table-striped' style='width:90%;'>
                    <tr><th>Popularity Range</th><th>Description</th></tr>
                    <tr><td>0.76-1.0</td><td>Very Popular</td></tr>
                    <tr><td>0.51-0.75</td><td>Popular</td></tr>
                    <tr><td>0.26-0.50</td><td>Moderate</td></tr>
                    <tr><td>0-0.25</td><td>Less Popular</td></tr>
                  </table>"""
    else:  # For songs
        legend = """<table class='table table-striped' style='width:90%;'>
                    <tr><th>Similarity Score</th><th>Description</th></tr>
                    <tr><td>0.8-1.0</td><td>Highly Similar</td></tr>
                    <tr><td>0.6-0.79</td><td>Moderately Similar</td></tr>
                    <tr><td>0.4-0.59</td><td>Slightly Similar</td></tr>
                    <tr><td>0-0.39</td><td>Low Similarity</td></tr>
                  </table>"""

    return f"{recommended_songs_output_html}<br><br>{legend}<br><br><b>Similarity-based Precision:</b> {precision:.2f}", ""

# Function to load the latest dataset version
def load_latest_dataset():
    versions = [f for f in os.listdir("data_versions/") if f.startswith("data_") and f.endswith(".csv")]
    if not versions:
        raise FileNotFoundError("No dataset versions found.")

    # Sort by timestamp in filename
    versions.sort(reverse=True)
    latest_version = versions[0].split("_")[1].replace(".csv", "")
    df, metadata = load_dataset_version(latest_version)

    print(f"Loaded dataset version {latest_version} with metadata: {metadata}")
    return df

def main():
    global df, features, current_df
    file_path = 'data.csv'

    # Clean and preprocess data
    df = clean_and_preprocess_data(file_path)

    # Define features
    features = ['danceability', 'energy', 'tempo', 'valence', 'loudness']

    # Initialize current dataset
    current_size = 100000
    increment = 10000
    current_df = df.head(current_size)

    # Launch GRadio Interface
    gr.Interface(
        fn=lambda search_artist, search_song, tolerance, current_size, increment: gradio_interface(
            search_artist.strip(), search_song.strip(), tolerance, current_size, increment
        ),
        inputs=[
            gr.Textbox(label="Enter Artist Name (Optional)"),
            gr.Textbox(label="Enter Song Name (Optional)"),
            gr.Slider(minimum=0.05, maximum=0.2, step=0.05, label="Tolerance (%)", value=0.1),
            gr.Number(label="Current Data Size", value=current_size),
            gr.Number(label="Increment Size", value=increment),
        ],
        outputs=[
            gr.HTML(label="Recommendations"),
            gr.HTML(label="Legend")
        ],
        title="Spotify Recommender",
        description="Enter either an artist's name or a song name to get song recommendations and explanations for the recommendations. Adjust the dataset size dynamically."
    ).launch(share=True)

if __name__ == "__main__":
    main()
