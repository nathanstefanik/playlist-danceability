import numpy as np
import pandas as pd

print('Working...')
filepath = 'src/tracks_numerical_features.pkl'
tracks_numerical_features = pd.read_csv('tracks_features.csv')
tracks_numerical_features = tracks_numerical_features[['id', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']]
tracks_numerical_features['explicit'] = tracks_numerical_features['explicit'].apply(lambda i: 1 if i else 0)
tracks_numerical_features.to_pickle(filepath)
print(f"Saved tracks' numerical features dataframe to {filepath}")