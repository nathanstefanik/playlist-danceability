import numpy as np
import pandas as pd

X = np.load('crafted_data/X.npy')
y = np.load('crafted_data/y.npy')
tracks_features = pd.read_pickle('src/tracks_numerical_features.pkl')

print('\n--- original ranges ---')
for col in tracks_features.columns[1:]:
    print(f'{col} range: {np.min(tracks_features[col].values)} to {np.max(tracks_features[col].values)}')
haha = tracks_features.copy()
def absMinMax(arr):
    mi = np.min(arr)
    ma = np.max(arr)
    return (arr - mi) / (ma - mi)
for col in haha.columns[1:]:
    haha[col] = absMinMax(haha[col].values.flatten())

haha.to_pickle('src/normalized_tracks_numerical_features.pkl')

print('--- DONE ---\n')