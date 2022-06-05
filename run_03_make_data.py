import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle
import data_utils
import os
import numpy as np
import data_utils

with open('data/guitar_and_synth_audio.pickle', 'rb') as handle:
    audios = pickle.load(handle)

# size in samples
sz = 256
w = np.hanning( sz )

# XY matrices
# how many
N = len(audios[0]['guitar'])//sz
x = np.zeros( (N*len( audios ) , sz) ).astype(np.float32)
y = np.zeros( (N*len( audios ) , sz) ).astype(np.float32)

print('x.shape: ', x.shape)

row = 0
for i, p in enumerate( audios ):
    if i%100 == 0:
        print(str(i) + ' / ' + str(len(audios)))
    ii = 0
    while ii + sz <= p['guitar'].size:
        x[row,:] = w*p['guitar'][ii:ii+sz].astype(np.float32)
        y[row,:] = w*p['synth'][ii:ii+sz].astype(np.float32)
        row += 1
        ii += sz


with open('data/x.pickle', 'wb') as handle:
    pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/y.pickle', 'wb') as handle:
    pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)