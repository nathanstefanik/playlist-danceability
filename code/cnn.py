# imports
import numpy as np
import pandas as pd

# Set these parameters
want_new_data = True

if want_new_data:
    data = pd.read_pickle('crafted_data/playlists_and_trackid_start0_end6900.pkl')
    data['num_tracks_id'] = data['tracks'].apply(lambda row: len(row))
    data = data[data['num_tracks_id'] >= 20]
    max_length = max(data['num_tracks_id'])

    numerical_values = pd.read_pickle('src/tracks_numerical_features.pkl')
    numerical_values = numerical_values.set_index('id')
    X = []
    y = []
    for playlist in data['tracks']:
        curr = []
        for i in range(len(playlist)):
            
            song_id = playlist[i]
            if (i == len(playlist) - 1):
                y.append(np.array(numerical_values.loc[song_id].values))
            else:
                curr.append(np.array(numerical_values.loc[song_id].values))
        to_pad = max_length - len(curr) - 1
        tmp = np.array(curr)
        tmp = np.pad(tmp, pad_width=((to_pad,0), (0,0)), mode='constant')
        X.append(np.array(tmp))
    X = np.array(X)
    y = np.array(y)
    np.save('crafted_data/X', X)
    np.save('crafted_data/y', y)
else:
    X = np.load('crafted_data/X.npy')
    y = np.load('crafted_data/y.npy')

print('X shape: ', X.shape)
print('y shape: ', y.shape)