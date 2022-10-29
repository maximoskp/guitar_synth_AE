from gettext import translation
from statistics import mode
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

with open('data/x.pickle', 'rb') as handle:
    x = pickle.load(handle)

with open('data/y.pickle', 'rb') as handle:
    y = pickle.load(handle)

sz = x.shape[1]

idxs = np.random.permutation( x.shape[0] )

x = x[idxs[:len(idxs)//10],:]
y = y[idxs[:len(idxs)//10],:]


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

input_shape = [sz,1]

batch_size = 64
dataset = tf.data.Dataset.from_tensor_slices( ( x_train , y_train ) )
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

# # build test images
# os.makedirs( 'figs', exist_ok=True )
# for i in range(10):
#     plt.clf()
#     plt.subplot(2,1,1)
#     plt.plot(x_train[i,:,:])
#     plt.subplot(2,1,2)
#     plt.plot(y_train[i,:,:])
#     plt.savefig('figs/train'+str(i)+'.png', dpi=300)


# guitar AE
# create the model
guitar_encoder = keras.models.Sequential()
guitar_encoder.add(keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
guitar_encoder.add(keras.layers.Conv1D(32, kernel_size=3, activation='relu'))
guitar_encoder.add(keras.layers.Conv1D(16, kernel_size=3, activation='relu'))
guitar_encoder.add(keras.layers.Conv1D(8, kernel_size=3, activation='relu'))

guitar_encoder.summary()

guitar_decoder = keras.models.Sequential()
guitar_decoder.add(keras.layers.Conv1DTranspose(8, kernel_size=3, activation='relu'))
guitar_decoder.add(keras.layers.Conv1DTranspose(16, kernel_size=3, activation='relu'))
guitar_decoder.add(keras.layers.Conv1DTranspose(32, kernel_size=3, activation='relu'))
guitar_decoder.add(keras.layers.Conv1DTranspose(64, kernel_size=3, activation='relu'))
guitar_decoder.add(keras.layers.Conv1D(1, kernel_size=3, activation='tanh', padding='same'))

# synth AE
# create the model
synth_encoder = keras.models.Sequential()
synth_encoder.add(keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
synth_encoder.add(keras.layers.Conv1D(32, kernel_size=3, activation='relu'))
synth_encoder.add(keras.layers.Conv1D(16, kernel_size=3, activation='relu'))
synth_encoder.add(keras.layers.Conv1D(8, kernel_size=3, activation='relu'))

synth_decoder = keras.models.Sequential()
synth_decoder.add(keras.layers.Conv1DTranspose(8, kernel_size=3, activation='relu'))
synth_decoder.add(keras.layers.Conv1DTranspose(16, kernel_size=3, activation='relu'))
synth_decoder.add(keras.layers.Conv1DTranspose(32, kernel_size=3, activation='relu'))
synth_decoder.add(keras.layers.Conv1DTranspose(64, kernel_size=3, activation='relu'))
synth_decoder.add(keras.layers.Conv1D(1, kernel_size=3, activation='tanh', padding='same'))

# translator
translator = keras.models.Sequential()
translator.add(keras.layers.Flatten(input_shape=(248,8)))
translator.add(keras.layers.Dense(256, activation='relu'))
translator.add(keras.layers.Dense(256, activation='relu'))
translator.add(keras.layers.Dense(248*8, activation='relu'))
translator.add(keras.layers.Reshape( (248,8) ))

guitar_model = keras.models.Sequential([guitar_encoder, guitar_decoder])
synth_model = keras.models.Sequential([synth_encoder, synth_decoder])
model = keras.models.Sequential([guitar_encoder, translator, synth_decoder])

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

os.makedirs( 'models', exist_ok=True )

filepath = 'models/synth_CNN_epoch{epoch:02d}_valLoss{val_loss:.6f}.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')

filepath_current_best = 'models/synth_CNN_current_best.hdf5'
checkpoint_current_best = ModelCheckpoint(filepath=filepath_current_best,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')

if os.path.exists('/models/guitar_synth_logger.csv'):
    os.remove('/models/guitar_synth_logger.csv')
csv_logger = CSVLogger('models/guitar_synth_logger.csv', append=True, separator=';')

guitar_model.compile(optimizer='adam', loss='mean_squared_error')
synth_model.compile(optimizer='adam', loss='mean_squared_error')
model.compile(optimizer='adam', loss='mean_squared_error')

guitar_encoder.summary()
synth_decoder.summary()
model.summary()

# history = model.fit(x_train, y_train, batch_size=256, epochs=1000, validation_data=(x_valid, y_valid), callbacks=[checkpoint, checkpoint_current_best, csv_logger])

import progressbar

def train_model(model, guitar_model, synth_model, x, y, batch_size=128, n_epochs=1000):
    for epoch in range(n_epochs):
        print('epoch: ', epoch)
        progress = 0.
        string2print = 'Epoch ' + str(epoch) + ': ' + 'gutar: ' + ' X ' + ' - synth: ' + ' X' + ' - model: ' + ' X' + ' |'
        widgets = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ', progressbar.FormatLabel( string2print ) ]
        widgets = [progressbar.FormatLabel(''), ' ', progressbar.Percentage(), ' ', progressbar.Bar('/'), ' ', progressbar.RotatingMarker()]
        bar = progressbar.ProgressBar(maxval=1, widgets=widgets)
        bar.start()
        for x_batch, y_batch in dataset:
            progress += 1/dataset.cardinality().numpy()
            progress = min( progress , 1 )
            # 1 - train guitar
            # guitar_model.trainable = True # break out encoder-decoder and switch encoder trainability
            # print('training guitar AE')
            # guitar_error = guitar_model.fit(x, x, batch_size=batch_size , epochs=1)
            guitar_error = guitar_model.train_on_batch(x_batch, x_batch)
            # 2 - train synth
            # synth_model.trainable = True # break out encoder-decoder and switch decode trainability
            # print('training synth AE')
            # synth_error = synth_model.fit(y, y, batch_size=batch_size , epochs=1)
            synth_error = synth_model.train_on_batch(y_batch, y_batch)
            # 3 train model
            # print('training model')
            # model_error = model.fit(x, y, batch_size=batch_size, epochs=1)
            model_error = synth_model.train_on_batch(x_batch, y_batch)
            string2print = 'Epoch ' + str(epoch) + '| ' + 'guitar: ' + '{:.9f}'.format(guitar_error) + ' - synth: ' + '{:.9f}'.format(synth_error)  + ' - model: ' + '{:.6f}'.format(model_error)
            widgets[4] = progressbar.FormatLabel( string2print )
            bar.update(progress)
        # print( 'epoch: ' + str(epoch) + ' - discriminator: ' + str(discr_error) + ' - gan: ' + str(gan_error) )
        # d_pred = discriminator.predict( X_train )
        # d_error = tf.metrics.binary_crossentropy( y_train , d_pred )
        # g_pred = gan.predict( X_train )
        # g_error = tf.metrics.binary_crossentropy( y_train , d_pred )
        # string2print = 'Epoch ' + str(epoch) + '| ' + 'd: ' + '{:.4f}'.format(d_error) + ' - g: ' + '{:.4f}'.format(gan_error)
        # widgets[4] = progressbar.FormatLabel( string2print )
        # bar.update(progress)
        # bar.finish()

train_model( model, guitar_model, synth_model, x_train, y_train )
