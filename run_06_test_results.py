import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle
import data_utils
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

with open('data/y_sine.pickle', 'rb') as handle:
    x = pickle.load(handle)

with open('data/guitar_and_synth_audio.pickle', 'rb') as handle:
    audios = pickle.load(handle)

sz = x.shape[1]

# load model
model = keras.models.load_model( 'models/synth_AE/synth_CNN_current_best.hdf5' )

amplitude = np.iinfo(np.int16).max

# ===================================================================================
# sine tests
# ===================================================================================
print('sine tests ===========================')
folder_name = 'sine_sims'

os.makedirs( 'audios', exist_ok=True )
os.makedirs( 'audios/' + folder_name, exist_ok=True )

os.makedirs( 'figs', exist_ok=True )
os.makedirs( 'figs/' + folder_name, exist_ok=True )

num_sims = 10
idxs = np.random.randint( x.shape[0], size=(num_sims) )
for i in range(num_sims):
    print('running' + str(i+1) + '/' + str(num_sims))
    x_in = x[idxs[i],:]
    y = np.squeeze( model.predict( np.reshape( x_in, (1,sz,1) ) ) )
    plt.clf()
    plt.subplot(2,1,1)
    plt.plot(x_in)
    plt.subplot(2,1,2)
    plt.plot(y)
    plt.savefig('figs/' + folder_name + '/sine_sim' + str(i) + '.png', dpi=300)
    write('audios/' + folder_name + '/sine_in' + str(i) + '.wav', 16000, (amplitude*x_in).astype(np.int16))
    write('audios/' + folder_name + '/sine_out' + str(i) + '.wav', 16000, (amplitude*y).astype(np.int16))


# ===================================================================================
# full sine tests
# ===================================================================================
print('full sine tests ===========================')
folder_name = 'full_sine_sims'

os.makedirs( 'audios', exist_ok=True )
os.makedirs( 'audios/' + folder_name, exist_ok=True )

os.makedirs( 'figs', exist_ok=True )
os.makedirs( 'figs/' + folder_name, exist_ok=True )

cross_fade_area = sz//2

num_sims = 10
idxs = np.random.randint( len(audios), size=(num_sims) )
for i in range(num_sims):
    print('running' + str(i+1) + '/' + str(num_sims))
    t = 0
    x_start = audios[idxs[i]]['sine']
    x_in_total = np.zeros( x_start.shape )
    y_total = np.zeros( x_start.shape )
    while t+sz < x_start.size:
        x_in = np.hanning( sz )*x_start[t:t+sz]
        y = np.squeeze( model.predict( np.reshape( x_in, (1,sz,1) ) ) )
        x_in_total[t:t+sz] = x_in_total[t:t+sz] + x_in
        y_total[t:t+sz] = y_total[t:t+sz] + y
        t += int(sz - cross_fade_area)
    plt.clf()
    plt.subplot(2,1,1)
    plt.plot(x_in_total)
    plt.subplot(2,1,2)
    plt.plot(y_total)
    plt.savefig('figs/' + folder_name + '/sine_sim' + str(i) + '.png', dpi=300)
    write('audios/' + folder_name + '/sine_in' + str(i) + '.wav', 16000, (amplitude*x_in_total).astype(np.int16))
    write('audios/' + folder_name + '/sine_out' + str(i) + '.wav', 16000, (amplitude*y_total).astype(np.int16))

# ===================================================================================
# full guitar tests
# ===================================================================================
print('full guitar tests ===========================')
folder_name = 'full_guitar_sims'

os.makedirs( 'audios', exist_ok=True )
os.makedirs( 'audios/' + folder_name, exist_ok=True )

os.makedirs( 'figs', exist_ok=True )
os.makedirs( 'figs/' + folder_name, exist_ok=True )

cross_fade_area = sz//2

num_sims = 10
idxs = np.random.randint( len(audios), size=(num_sims) )
for i in range(num_sims):
    print('running' + str(i+1) + '/' + str(num_sims))
    t = 0
    x_start = audios[idxs[i]]['guitar']
    x_in_total = np.zeros( x_start.shape )
    y_total = np.zeros( x_start.shape )
    while t+sz < x_start.size:
        x_in = np.hanning( sz )*x_start[t:t+sz]
        y = np.squeeze( model.predict( np.reshape( x_in, (1,sz,1) ) ) )
        x_in_total[t:t+sz] = x_in_total[t:t+sz] + x_in
        y_total[t:t+sz] = y_total[t:t+sz] + y
        t += int(sz - cross_fade_area)
    plt.clf()
    plt.subplot(2,1,1)
    plt.plot(x_in_total)
    plt.subplot(2,1,2)
    plt.plot(y_total)
    plt.savefig('figs/' + folder_name + '/sine_sim' + str(i) + '.png', dpi=300)
    write('audios/' + folder_name + '/guitar_in' + str(i) + '.wav', 16000, (amplitude*x_in_total).astype(np.int16))
    write('audios/' + folder_name + '/guitar_out' + str(i) + '.wav', 16000, (amplitude*y_total).astype(np.int16))