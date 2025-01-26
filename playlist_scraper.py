import spotipy
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
import streamlit as st
import pandas as pd

st.set_page_config(page_title='Spotify Recommender', page_icon=':musical_note:')

CLIENT_ID = 'dc22ddcc83564d51830e72dc175498bf'
CLIENT_SECRET = 'f7f0eac9100c449398e5fef9a350369e'
REDIRECT_URI = 'http://localhost:5000'

sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope="user-read-private,user-read-email,user-library-read,playlist-read-private,streaming, user-read-playback-state, user-modify-playback-state,user-read-currently-playing, playlist-read-private, playlist-read-collaborative, playlist-modify-private, playlist-modify-public, user-top-read, user-read-recently-played",
        show_dialog=True
    )
)
st.title('Analysis for your Songs')
st.write('Discover insights about your Spotify listening habits.')

playlist_id = '1Ndsgv4FTKG1KwML9cGZxD'
playlist_data = {}
offset=0
while(sp.playlist_tracks(playlist_id=playlist_id,offset=offset) and offset < 400):
    playlist_data.update(sp.playlist_tracks(playlist_id=playlist_id,offset=offset))
    offset+=100

print(playlist_data)
tracks = playlist_data['items']
track_ids = [track['track']['id'] for track in tracks]
track_names = [track['track']['name'] for track in tracks]

# audio_features = sp.audio_features(track_ids)

# df = pd.DataFrame(audio_features)
# df['track_name'] = track_names
# df = df[['track_name', 'danceability', 'energy', 'valence']]
# df.set_index('track_name', inplace=True)

# st.subheader('Audio Features for Tracks in Playlist')
# st.bar_chart(df, height=500)
