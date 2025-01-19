import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from agent_app import clean_and_preprocess_data

def test_clean_and_preprocess_data_removes_nulls():
    data = {
        "name": ["Song 1", "Song 2", "Song 3"],
        "tempo": [120, None, 130],
        "valence": [0.5, None, 0.6],
        "popularity": [70, 80, None],
    }
    df = pd.DataFrame(data)

    processed_df = clean_and_preprocess_data(df=df)

    # Provjeri da nema nedostajućih vrijednosti
    assert processed_df.isnull().sum().sum() == 0, "There are still missing values in the processed DataFrame"

def test_clean_and_preprocess_data_normalization():
    data = {
        "name": ["Song 1", "Song 2"],
        "tempo": [120, 130],
        "valence": [0.5, 0.7],
        "popularity": [70, 80],
    }
    df = pd.DataFrame(data)
    processed_df = clean_and_preprocess_data(df=df)
    for col in ["tempo", "valence", "popularity"]:
        assert processed_df[col].min() == 0  # Normalizirani minimum
        assert processed_df[col].max() == 1  # Normalizirani maksimum

