import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import pandas as pd
from agent_app import clean_and_preprocess_data, generate_recommendations

def test_clean_and_preprocess_data():
    data = {
        "name": ["Song 1", "Song 2", "Song 3"],
        "tempo": [120, None, 130],
        "valence": [0.5, 0.7, 0.6],
        "popularity": [70, 80, None],
        "danceability": [0.8, 0.9, None],
        "energy": [0.7, 0.6, 0.8],
    }
    df = pd.DataFrame(data)

    # Test funkcije
    processed_df = clean_and_preprocess_data(df=df)

    # Provjeri da nema null vrijednosti
    assert processed_df.isnull().sum().sum() == 0

    # Provjeri da su duplicirani redovi uklonjeni
    assert processed_df['name'].duplicated().sum() == 0

def test_generate_recommendations():
    data = {
        "name": ["Song 1", "Song 2", "Song 3"],
        "artists": ["Artist A", "Artist B", "Artist C"],
        "tempo": [120, 110, 115],
        "valence": [0.5, 0.6, 0.7],
        "popularity": [0.8, 0.7, 0.9],
    }
    df = pd.DataFrame(data)
    recommended, _, _, _ = generate_recommendations(df, search_artist="Artist A", search_song=None, features=["tempo", "valence"])
    assert not recommended.empty
