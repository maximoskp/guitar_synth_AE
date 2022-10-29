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

with open('data/y_sine.pickle', 'rb') as handle:
    x = pickle.load(handle)

with open('data/y_saw.pickle', 'rb') as handle:
    y = pickle.load(handle)

sz = x.shape[1]

idxs = np.random.permutation( x.shape[0] )

x = x[idxs[:len(idxs)//5],:]
y = y[idxs[:len(idxs)//5],:]


rnd_idxs = np.random.permutation( x.shape[0] )
# # make sure noise and empty are first
# rnd_idxs = np.insert(rnd_idxs, 0 , rnd_idxs.size-1)
# rnd_idxs = np.insert(rnd_idxs, 0 , rnd_idxs.size-3)

tr = 2*x.shape[0]//3
v = x.shape[0]//6
te = x.shape[0]//6

x_train = np.expand_dims( x[ rnd_idxs[:tr] ,:], axis=2 )
y_train = np.expand_dims( y[ rnd_idxs[:tr] ,:], axis=2 )

x_valid = np.expand_dims( x[ rnd_idxs[tr:tr+v] ,:], axis=2 )
y_valid = np.expand_dims( y[ rnd_idxs[tr:tr+v] ,:], axis=2 )

x_test = np.expand_dims( x[ rnd_idxs[tr+v:tr+v+te] ,:], axis=2 )
y_test = np.expand_dims( y[ rnd_idxs[tr+v:tr+v+te] ,:], axis=2 )

max_norm_value = 2.0
input_shape = [sz,1]

# build test images
os.makedirs( 'figs', exist_ok=True )
for i in range(10):
    plt.clf()
    plt.subplot(2,1,1)
    plt.plot(x_train[i,:,:])
    plt.subplot(2,1,2)
    plt.plot(y_train[i,:,:])
    plt.savefig('figs/train'+str(i)+'.png', dpi=300)


# create the model
encoder = keras.models.Sequential()
encoder.add(keras.layers.Conv1D(32, kernel_size=15, strides=1, padding='same', activation='relu', input_shape=input_shape))
encoder.add(keras.layers.MaxPool1D(pool_size=4, padding='valid'))
encoder.add(keras.layers.Conv1D(64, kernel_size=15, strides=1, padding='same', activation='relu'))
encoder.add(keras.layers.MaxPool1D(pool_size=4, padding='valid'))
encoder.add(keras.layers.Conv1D(128, kernel_size=15, strides=1, padding='same', activation='relu'))
encoder.add(keras.layers.MaxPool1D(pool_size=4, padding='valid'))
encoder.add(keras.layers.Conv1D(256, kernel_size=15, strides=1, padding='same', activation='relu'))
encoder.add(keras.layers.MaxPool1D(pool_size=4, padding='valid'))
encoder.add(keras.layers.Conv1D(512, kernel_size=15, strides=1, padding='same', activation='relu'))
encoder.add(keras.layers.MaxPool1D(pool_size=4, padding='valid'))
encoder.add(keras.layers.Conv1D(1024, kernel_size=15, strides=1, padding='same', activation='relu'))
# encoder.add(keras.layers.Conv1D(64, kernel_size=3, kernel_constraint=keras.constraints.max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
# encoder.add(keras.layers.Conv1D(32, kernel_size=3, kernel_constraint=keras.constraints.max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
# encoder.add(keras.layers.Conv1D(16, kernel_size=3, kernel_constraint=keras.constraints.max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
# encoder.add(keras.layers.Conv1D(8, kernel_size=3, kernel_constraint=keras.constraints.max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))

decoder = keras.models.Sequential()
# decoder.add(keras.layers.Conv1DTranspose(8, kernel_size=3, kernel_constraint=keras.constraints.max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
# decoder.add(keras.layers.Conv1DTranspose(16, kernel_size=3, kernel_constraint=keras.constraints.max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
# decoder.add(keras.layers.Conv1DTranspose(32, kernel_size=3, kernel_constraint=keras.constraints.max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
# decoder.add(keras.layers.Conv1DTranspose(64, kernel_size=3, kernel_constraint=keras.constraints.max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
# decoder.add(keras.layers.Conv1D(1, kernel_size=3, kernel_constraint=keras.constraints.max_norm(max_norm_value), activation='tanh', padding='same'))
decoder.add(keras.layers.Conv1D(1024, kernel_size=5, strides=1, activation='relu', padding='same'))
decoder.add(keras.layers.UpSampling1D(size=4))
decoder.add(keras.layers.Conv1D(512, kernel_size=5, strides=1, activation='relu', padding='same'))
decoder.add(keras.layers.UpSampling1D(size=4))
decoder.add(keras.layers.Conv1D(256, kernel_size=5, strides=1, activation='relu', padding='same'))
decoder.add(keras.layers.UpSampling1D(size=4))
decoder.add(keras.layers.Conv1D(128, kernel_size=5, strides=1, activation='relu', padding='same'))
decoder.add(keras.layers.UpSampling1D(size=4))
decoder.add(keras.layers.Conv1D(64, kernel_size=5, strides=1, activation='relu', padding='same'))
decoder.add(keras.layers.UpSampling1D(size=4))
decoder.add(keras.layers.Conv1D(32, kernel_size=5, strides=1, activation='relu', padding='same'))
decoder.add(keras.layers.Conv1D(1, kernel_size=5, strides=1, activation='tanh', padding='same'))

model = keras.models.Sequential([encoder, decoder])

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

os.makedirs( 'models', exist_ok=True )
os.makedirs( 'models/synth_AE', exist_ok=True )

filepath = 'models/synth_AE/synth_CNN_epoch{epoch:02d}_valLoss{val_loss:.6f}.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')

filepath_current_best = 'models/synth_AE/synth_CNN_current_best.hdf5'
checkpoint_current_best = ModelCheckpoint(filepath=filepath_current_best,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')

if os.path.exists('/models/synth_AE/full_tab_logger.csv'):
    os.remove('/models/synth_AE/full_tab_logger.csv')
csv_logger = CSVLogger('models/synth_AE/full_tab_logger.csv', append=True, separator=';')

encoder.summary()
decoder.summary()
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(x_train, y_train, batch_size=32, epochs=1000, validation_data=(x_valid, y_valid), callbacks=[checkpoint, checkpoint_current_best, csv_logger])