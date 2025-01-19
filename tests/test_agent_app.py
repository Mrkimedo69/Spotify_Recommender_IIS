import pytest
import pandas as pd
from agent_app import clean_and_preprocess_data, generate_recommendations
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_clean_and_preprocess_data():
    data = {
        "name": ["Song 1", "Song 2"],
        "tempo": [120, None],
        "valence": [0.5, 0.7],
        "popularity": [70, None],
    }
    df = pd.DataFrame(data)
    processed_df = clean_and_preprocess_data(df)
    assert processed_df.isnull().sum().sum() == 0  # Ensure no missing values

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
