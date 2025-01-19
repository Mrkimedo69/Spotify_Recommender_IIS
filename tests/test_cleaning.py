import pytest
import pandas as pd
from agent_app import clean_and_preprocess_data

def test_clean_and_preprocess_data_removes_nulls():
    data = {
        "name": ["Song 1", "Song 2", "Song 3"],
        "tempo": [120, None, 130],
        "valence": [0.5, 0.6, None],
    }
    df = pd.DataFrame(data)
    processed_df = clean_and_preprocess_data(df)
    assert processed_df.isnull().sum().sum() == 0

def test_clean_and_preprocess_data_normalization():
    data = {
        "name": ["Song 1", "Song 2"],
        "tempo": [120, 130],
        "valence": [0.5, 0.7],
        "popularity": [70, 80],
    }
    df = pd.DataFrame(data)
    processed_df = clean_and_preprocess_data(df)
    assert (processed_df['tempo'] <= 1).all() and (processed_df['tempo'] >= 0).all()
    assert (processed_df['popularity'] <= 1).all() and (processed_df['popularity'] >= 0).all()
