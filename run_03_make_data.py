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
# sz = 256
# keep entire
sz = 4096
w = np.hanning( sz )

# XY matrices
# how many
N = len(audios[0]['guitar'])//sz
x = np.zeros( (N*len( audios ) , sz) ).astype(np.float32)
y_sine = np.zeros( (N*len( audios ) , sz) ).astype(np.float32)
y_saw = np.zeros( (N*len( audios ) , sz) ).astype(np.float32)

print('x.shape: ', x.shape)

row = 0
for i, p in enumerate( audios ):
    if i%100 == 0:
        print(str(i) + ' / ' + str(len(audios)))
    ii = 0
    while ii + sz <= p['guitar'].size:
        x[row,:] = w*p['guitar'][ii:ii+sz].astype(np.float32)
        y_sine[row,:] = w*p['sine'][ii:ii+sz].astype(np.float32)
        y_saw[row,:] = w*p['saw'][ii:ii+sz].astype(np.float32)
        row += 1
        ii += sz


with open('data/x.pickle', 'wb') as handle:
    pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/y_sine.pickle', 'wb') as handle:
    pickle.dump(y_sine, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/y_saw.pickle', 'wb') as handle:
    pickle.dump(y_saw, handle, protocol=pickle.HIGHEST_PROTOCOL)