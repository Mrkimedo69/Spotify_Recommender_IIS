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
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore", message=".*LOKY_MAX_CPU_COUNT.*")
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

def clean_and_preprocess_data(file_path=None, df=None):
    if df is None and file_path is None:
        raise ValueError("Either 'file_path' or 'df' must be provided.")
    if df is None:
        df = pd.read_csv(file_path)

    missing_row_threshold = 0.5
    missing_col_threshold = 0.5
    df = df.loc[:, df.isnull().mean() < missing_col_threshold]
    df = df[df.isnull().mean(axis=1) < missing_row_threshold]

    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].mean())

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('Unknown')

    if 'name' in df.columns:
        df = df.drop_duplicates(subset=['name'], keep='first').reset_index(drop=True)

    min_max_features = ['tempo', 'valence', 'popularity']
    z_score_features = ['danceability', 'energy']
    log_transform_features = ['loudness']

    existing_min_max_features = [col for col in min_max_features if col in df.columns]
    if existing_min_max_features:
        scaler_min_max = MinMaxScaler()
        df[existing_min_max_features] = scaler_min_max.fit_transform(df[existing_min_max_features])

    existing_z_score_features = [col for col in z_score_features if col in df.columns]
    if existing_z_score_features:
        scaler_z_score = StandardScaler()
        df[existing_z_score_features] = scaler_z_score.fit_transform(df[existing_z_score_features])

    for feature in log_transform_features:
        if feature in df.columns:
            df[feature] = np.log1p(df[feature] - df[feature].min() + 1)

    return df

def evaluate_similarity_precision(df, playlist_indices, recommended_indices, features, tolerance=0.1):
    if not playlist_indices or not recommended_indices:
        return 0.0

    playlist_features = df.iloc[playlist_indices][features].mean().values
    recommended_features = df.iloc[recommended_indices][features].values

    similarities = cosine_similarity([playlist_features], recommended_features).flatten()

    relevant = (similarities >= (1 - tolerance)).astype(int)
    predicted = [1] * len(recommended_indices)

    return precision_score(relevant, predicted)

def generate_recommendations(df, search_artist, search_song, features, top_n=10):
    playlist_indices = []
    if search_artist:
        playlist_indices = df[df['artists'].str.contains(search_artist, case=False, na=False)].index.tolist()
    elif search_song:
        playlist_indices = df[df['name'].str.contains(search_song, case=False, na=False)].index.tolist()

    if not playlist_indices:
        return pd.DataFrame(), f"Sorry, no results found for '{search_artist or search_song}'.", [], []

    playlist_features = df.iloc[playlist_indices][features].mean(axis=0).values.reshape(1, -1)
    df_copy = df.copy()
    df_copy['similarity'] = cosine_similarity(playlist_features, df_copy[features]).flatten()
    df_copy = df_copy[~df_copy.index.isin(playlist_indices)]
    df_copy = df_copy.sort_values(by='similarity', ascending=False)
    recommended_songs = df_copy.head(top_n)
    explanations = [f"This song is recommended because of its similarity in {', '.join(features)}." for _ in range(len(recommended_songs))]
    return recommended_songs, "", playlist_indices, explanations

def generate_recommendations_hierarchical(df, search_artist, search_song, features, top_n=10):
    sample_size = 5000
    df_sample = df.sample(n=min(len(df), sample_size), random_state=42)

    if search_artist:
        additional_indices = df[df['artists'].str.contains(search_artist, case=False, na=False)].index.tolist()
    elif search_song:
        additional_indices = df[df['name'].str.contains(search_song, case=False, na=False)].index.tolist()
    else:
        additional_indices = []

    if additional_indices:
        additional_sample = df.loc[additional_indices]
        df_sample = pd.concat([df_sample, additional_sample]).drop_duplicates().reset_index(drop=True)

    n_components = min(10, len(features), len(df_sample))
    if n_components < 2:
        return pd.DataFrame(), "Insufficient dimensions for PCA.", [], []

    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df_sample[features])
    Z = linkage(df_pca, method='ward')
    df_sample['cluster'] = fcluster(Z, t=25, criterion='maxclust')

    playlist_indices = []
    if search_artist:
        playlist_indices = df_sample[df_sample['artists'].str.contains(search_artist, case=False, na=False)].index.tolist()
    elif search_song:
        playlist_indices = df_sample[df_sample['name'].str.contains(search_song, case=False, na=False)].index.tolist()

    if not playlist_indices:
        return pd.DataFrame(), f"Sorry, no results found for '{search_artist or search_song}'.", [], []

    search_cluster = df_sample.iloc[playlist_indices]['cluster'].values[0]
    recommended_songs = df_sample[df_sample['cluster'] == search_cluster].sort_values(by='popularity', ascending=False).head(top_n)
    explanations = [f"This song is recommended because it belongs to the same cluster based on features: {', '.join(features)}." for _ in range(len(recommended_songs))]
    return recommended_songs, "", playlist_indices, explanations

def generate_recommendations_tsne(df, search_artist, search_song, features, top_n=10):
    sample_size = 1000
    df_sample = df.sample(n=min(len(df), sample_size), random_state=42)

    if search_artist:
        additional_indices = df[df['artists'].str.contains(search_artist, case=False, na=False)].index.tolist()
    elif search_song:
        additional_indices = df[df['name'].str.contains(search_song, case=False, na=False)].index.tolist()
    else:
        additional_indices = []

    if additional_indices:
        additional_sample = df.loc[additional_indices]
        df_sample = pd.concat([df_sample, additional_sample]).drop_duplicates().reset_index(drop=True)

    n_components = min(10, len(features), len(df_sample))
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(df_sample[features])

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=500)
    X_embedded = tsne.fit_transform(reduced_features)

    df_sample['tsne_x'] = X_embedded[:, 0]
    df_sample['tsne_y'] = X_embedded[:, 1]

    playlist_indices = []
    if search_artist:
        playlist_indices = df_sample[df_sample['artists'].str.contains(search_artist, case=False, na=False)].index.tolist()
    elif search_song:
        playlist_indices = df_sample[df_sample['name'].str.contains(search_song, case=False, na=False)].index.tolist()

    if not playlist_indices:
        return pd.DataFrame(), f"Sorry, no results found for '{search_artist or search_song}'.", [], []

    playlist_points = df_sample.iloc[playlist_indices][['tsne_x', 'tsne_y']].mean(axis=0).values
    df_sample['distance'] = np.linalg.norm(df_sample[['tsne_x', 'tsne_y']].values - playlist_points, axis=1)
    recommended_songs = df_sample.sort_values(by='distance').head(top_n)
    explanations = [f"This song is recommended because it is close in T-SNE space to your search." for _ in range(len(recommended_songs))]
    return recommended_songs, "", playlist_indices, explanations

def compare_methods(df, search_artist, search_song, features, top_n=10, tolerance=0.1):
    # Generiranje preporuka za sve metode
    recommended_cosine, _, cosine_indices, _ = generate_recommendations(df, search_artist, search_song, features, top_n)
    recommended_hierarchical, _, hierarchical_indices, _ = generate_recommendations_hierarchical(df, search_artist, search_song, features, top_n)
    recommended_tsne, _, tsne_indices, _ = generate_recommendations_tsne(df, search_artist, search_song, features, top_n)

    # Izračun preciznosti svake metode
    precision_cosine = evaluate_similarity_precision(
        df=df,
        playlist_indices=cosine_indices,
        recommended_indices=recommended_cosine.index.tolist(),
        features=features,
        tolerance=tolerance
    )
    precision_hierarchical = evaluate_similarity_precision(
        df=df,
        playlist_indices=hierarchical_indices,
        recommended_indices=recommended_hierarchical.index.tolist(),
        features=features,
        tolerance=tolerance
    )
    precision_tsne = evaluate_similarity_precision(
        df=df,
        playlist_indices=tsne_indices,
        recommended_indices=recommended_tsne.index.tolist(),
        features=features,
        tolerance=tolerance
    )

    # Izrada tablice usporedbe
    max_length = max(len(recommended_cosine), len(recommended_hierarchical), len(recommended_tsne))
    cosine_names = list(recommended_cosine['name'].values) if not recommended_cosine.empty else []
    hierarchical_names = list(recommended_hierarchical['name'].values) if not recommended_hierarchical.empty else []
    tsne_names = list(recommended_tsne['name'].values) if not recommended_tsne.empty else []

    cosine_names += [""] * (max_length - len(cosine_names))
    hierarchical_names += [""] * (max_length - len(hierarchical_names))
    tsne_names += [""] * (max_length - len(tsne_names))

    comparison_df = pd.DataFrame({
        'Cosine Similarity': cosine_names,
        'Hierarchical Clustering': hierarchical_names,
        'T-SNE': tsne_names
    })

    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Brisanje starog sadržaja prije pisanja novih logova
    log_file_path = os.path.join(logs_dir, "recommendation_logs.txt")
    with open(log_file_path, "w", encoding="utf-8") as log_file:  # "w" mode briše postojeći sadržaj
        log_file.write(f"\nSearch Artist: {search_artist}, Search Song: {search_song}\n")
        log_file.write("="*80 + "\n\n")

        # Zapis preporuka po metodama
        for method_name, recommendations in [
            ("Cosine Similarity", recommended_cosine),
            ("Hierarchical Clustering", recommended_hierarchical),
            ("T-SNE", recommended_tsne)
        ]:
            log_file.write(f"{method_name} Recommendations:\n")
            log_file.write("-"*50 + "\n")
            if not recommendations.empty:
                log_file.write(f"| {'Name'.ljust(30)} | {'Artists'.ljust(40)} | Popularity |\n")
                log_file.write("-"*50 + "\n")
                for _, row in recommendations.iterrows():
                    log_file.write(f"| {row['name'][:30].ljust(30)} | {', '.join(eval(row['artists']))[:40].ljust(40)} | {row['popularity']:.2f} |\n")
            else:
                log_file.write("No recommendations found.\n")
            log_file.write("\n")

        # Zapis tablice usporedbe
        log_file.write("Comparison Results:\n")
        log_file.write("-"*50 + "\n")
        log_file.write(comparison_df.to_string(index=False))
        log_file.write("\n" + "-"*80 + "\n")

        # Zapis preciznosti svake metode
        log_file.write("\nPrecision Scores:\n")
        log_file.write("-"*50 + "\n")
        log_file.write(f"Cosine Similarity Precision: {precision_cosine:.2f}\n")
        log_file.write(f"Hierarchical Clustering Precision: {precision_hierarchical:.2f}\n")
        log_file.write(f"T-SNE Precision: {precision_tsne:.2f}\n")
        log_file.write("\n" + "="*80 + "\n")

    return recommended_cosine, comparison_df

def gradio_interface(search_artist, search_song, tolerance, current_size, increment):
    global current_df, df

    if not search_artist and not search_song:
        return "<b>Please enter either an artist's name or a song name.</b>", ""

    if current_size > len(df):
        current_size = len(df)
    current_df = df.head(int(current_size))

    recommended_songs, rec_error, playlist_indices, explanations = generate_recommendations(
        current_df, search_artist, search_song, features, top_n=10
    )

    compare_methods(current_df, search_artist, search_song, features, top_n=10)

    if rec_error:
        return f"<b>{rec_error}</b>", ""

    recommended_songs['Explanation'] = explanations
    recommended_songs_output = recommended_songs[['name', 'artists', 'popularity', 'Explanation']].copy()
    recommended_songs_output['artists'] = recommended_songs_output['artists'].apply(lambda x: ', '.join(eval(x)))
    recommended_songs_output_html = recommended_songs_output.to_html(index=False, escape=False, classes='table table-striped')

    legend = """<table class='table table-striped' style='width:90%;'>
                    <tr><th>Popularity Range</th><th>Description</th></tr>
                    <tr><td>0.76-1.0</td><td>Very Popular</td></tr>
                    <tr><td>0.51-0.75</td><td>Popular</td></tr>
                    <tr><td>0.26-0.50</td><td>Moderate</td></tr>
                    <tr><td>0-0.25</td><td>Less Popular</td></tr>
                </table>"""

    return f"{recommended_songs_output_html}<br><br>{legend}", ""

def main():
    global df, features
    file_path = 'data.csv'
    df = clean_and_preprocess_data(file_path)
    features = ['danceability', 'energy', 'tempo', 'valence', 'loudness']

    gr.Interface(
        fn=lambda search_artist, search_song, tolerance, current_size, increment: gradio_interface(
            search_artist.strip(), search_song.strip(), tolerance, current_size, increment
        ),
        inputs=[
            gr.Textbox(label="Enter Artist Name (Optional)"),
            gr.Textbox(label="Enter Song Name (Optional)"),
            gr.Slider(minimum=0.01, maximum=0.2, step=0.01, label="Tolerance (%)", value=0.05),
            gr.Number(label="Current Data Size", value=100000),
            gr.Number(label="Increment Size", value=500),
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
