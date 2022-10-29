from re import T
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

os.makedirs('figs', exist_ok=True)
os.makedirs('figs/ir', exist_ok=True)

with open('data/x.pickle', 'rb') as handle:
    x = pickle.load(handle)

with open('data/y_sine.pickle', 'rb') as handle:
    y_sine = pickle.load(handle)

with open('data/y_saw.pickle', 'rb') as handle:
    y_saw = pickle.load(handle)

with open('data/guitar_and_synth_audio.pickle', 'rb') as handle:
    audios = pickle.load(handle)

sz = x.shape[1]

# load model
model = keras.models.load_model( 'models/synth_AE/synth_CNN_current_best.hdf5' )

# make impulse signal
imp_sig = np.zeros(sz)
imp_sig[100] = 1
# get impulse response
ir = np.squeeze( model.predict( np.reshape( imp_sig, (1,sz,1) ) ) )

ir = ir[:200]
# plot ir
plt.clf()
plt.plot(ir)
plt.savefig('figs/ir/ir.png',dpi=300)

# select a guitar signal
idx = 300
g = audios[idx]['sine']
print(g.shape)
print(ir.shape)
c = np.convolve(g,ir, 'same')
print(c.shape)

plt.clf()
plt.subplot(2,1,1)
plt.plot(g)
plt.subplot(2,1,2)
plt.plot(c)
plt.savefig('figs/ir/conv_test.png',dpi=300)

os.makedirs( 'audios', exist_ok=True )
amplitude = np.iinfo(np.int16).max
write("audios/guitar.wav", 16000, (amplitude*g).astype(np.int16))
write("audios/synth.wav", 16000, (amplitude*c).astype(np.int16))

# test phasor input
sine_in = y_sine[1000,:]
sine_out = np.squeeze( model.predict( np.reshape( sine_in, (1,sz,1) ) ) )

plt.clf()
plt.subplot(2,1,1)
plt.plot(sine_in)
plt.subplot(2,1,2)
plt.plot(sine_out)
plt.savefig('figs/ir/sine.png',dpi=300)

write("audios/sine_in.wav", 16000, (amplitude*sine_in).astype(np.int16))
write("audios/sine_out.wav", 16000, (amplitude*sine_out).astype(np.int16))