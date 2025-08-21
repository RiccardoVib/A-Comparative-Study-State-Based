# Copyright (C) 2025 Riccardo Simionato, University of Oslo
# Inquiries: riccardo.simionato.vib@gmail.com.com
#
# This code is free software: you can redistribute it and/or modify it under the terms
# of the GNU Lesser General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Less General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this code.
# If not, see <http://www.gnu.org/licenses/>.
#
# If you use this code or any part of it in any program or publication, please acknowledge
# its authors by adding a reference to this publication:
#
# R. Simionato, 2025, "A Comparative Study of State-based Neural Networks for Virtual Analog Audio Effects Modeling" in EURASIP Journal on Audio, Speech, and Music Processing, 2025.

import os
import tensorflow as tf
from UtilsForTrainings import plotTraining, writeResults, checkpoints, predictWaves, MyLRScheduler
from Models import create_model_S4D, create_model_LSTM, create_model_ED, create_model_LRU, create_model_S6
from DatasetsClass import DataGeneratorPickles
import numpy as np
#import random
from Metrics import ESR, NRMSE, STFT, SF
import sys
import time
#import matplotlib.pyplot as plt


def train(**kwargs):
    batch_size = kwargs.get('batch_size', 1)
    mini_batch_size = kwargs.get('mini_batch_size', 1)
    learning_rate = kwargs.get('learning_rate', 1e-1)
    units = kwargs.get('units', 16)
    model_save_dir = kwargs.get('model_save_dir', '../../TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    inference = kwargs.get('inference', False)
    dataset = kwargs.get('dataset', None)
    model_name = kwargs.get('model', None)
    data_dir = kwargs.get('data_dir', '../../../Files/')
    epochs = kwargs.get('epochs', 60)

    start = time.time()

    #####seed
    #np.random.seed(42)
    #tf.random.set_seed(42)
    #random.seed(42)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    fs = 48000
    if dataset == 'FilterNeutron':
        data_dir = data_dir + 'FilterNeutron'
        D = 2
    elif dataset == 'CL1B':
        data_dir = data_dir + 'CL1B'
        D = 4
    elif dataset == 'LA2A':
        data_dir = data_dir + 'LA2A'
        D = 2
        fs = 44100
    elif dataset == 'OD':
        data_dir = data_dir + 'OD'
        D = 2
    elif dataset == 'Saturator':
        data_dir = data_dir + 'Saturator'
        D = 1
    elif dataset == 'ODNeutron':
        data_dir = data_dir + 'ODNeutron'
        D = 2
    elif dataset == 'Pultec':
        data_dir = data_dir + 'Pultec'
        D = 2
    else:
        data_dir = None

    w = 64
    if model_name == 'LRU':
        model = create_model_LRU(cond_dim=D, input_dim=w, units=units + 4, mini_batch_size=mini_batch_size,
                                 b_size=batch_size, stateful=False)
    elif model_name == 'S4D':
        model = create_model_S4D(cond_dim=D, input_dim=w, units=units + 4, mini_batch_size=mini_batch_size,
                                 b_size=batch_size, stateful=True)
    elif model_name == 'S6':
        model = create_model_S6(cond_dim=D, input_dim=w, units=units + 4, mini_batch_size=mini_batch_size,
                                b_size=batch_size, stateful=True)
    elif model_name == 'LSTM':
        model = create_model_LSTM(cond_dim=D, input_dim=w, units=units, mini_batch_size=mini_batch_size,
                                  b_size=batch_size, stateful=True)
    elif model_name == 'ED':
        model = create_model_ED(cond_dim=D, input_dim=w, units=units, mini_batch_size=mini_batch_size,
                                b_size=batch_size, stateful=True)
    else:
        model = None

    callbacks = []
    ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(model_save_dir, save_folder)
    test_gen = DataGeneratorPickles(data_dir, dataset + '_test1.pickle', input_size=w, cond=D,
                                    mini_batch_size=mini_batch_size, batch_size=batch_size)

    if not inference:
        callbacks += [ckpt_callback, ckpt_callback_latest]
        last = tf.train.latest_checkpoint(ckpt_dir_latest)
        if last is not None:
            print("Restored weights from {}".format(ckpt_dir_latest))
            model.load_weights(last)

        else:
            print("Initializing random weights.")

        # load the datasets
        train_gen = DataGeneratorPickles(data_dir, dataset + '_train.pickle', input_size=w, cond=D,
                                         mini_batch_size=mini_batch_size, batch_size=batch_size)

        training_steps = train_gen.training_steps
        opt = tf.keras.optimizers.Adam(learning_rate=MyLRScheduler(learning_rate, training_steps, epochs), clipnorm=1)
        model.compile(loss='mse', optimizer=opt)

        loss_training = np.empty(epochs)
        loss_val = np.empty(epochs)
        best_loss = 1e9
        count = 0
        for i in range(epochs):
            start = time.time()
            print('epochs:', i)
            model.reset_states()
            if model_name in ['S4D', 'LRU', 'S6']:
                model.layers[3].reset_states()

            print(model.optimizer.learning_rate)

            results = model.fit(train_gen, epochs=1, verbose=0, shuffle=False, validation_data=test_gen,
                                callbacks=callbacks)
            loss_training[i] = results.history['loss'][-1]
            loss_val[i] = results.history['val_loss'][-1]
            print(results.history['val_loss'][-1])

            if results.history['val_loss'][-1] < best_loss:
                best_loss = results.history['val_loss'][-1]
                count = 0
            else:
                count = count + 1
                if count == 20:
                    break
            avg_time_epoch = (time.time() - start)

            sys.stdout.write(f" Average time/epoch {'{:.3f}'.format(avg_time_epoch / 60)} min")
            sys.stdout.write("\n")

        # save results
        writeResults(results, units, epochs, batch_size, learning_rate, model_save_dir,
                     save_folder, epochs)

        loss_training = np.array(loss_training[:i])
        loss_val = np.array(loss_val[:i])
        plotTraining(loss_training, loss_val, model_save_dir, save_folder, str(epochs))

        print("Training done")

    avg_time_epoch = (time.time() - start)
    sys.stdout.write(f" Average time training{'{:.3f}'.format(avg_time_epoch / 60)} min")
    sys.stdout.write("\n")
    sys.stdout.flush()

    best = tf.train.latest_checkpoint(ckpt_dir)
    if best is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(best).expect_partial()
    else:
        print("Weights not found. Using random weights.")

    model.reset_states()
    if model_name in ['S4D', 'LRU', 'S6']:
        model.layers[3].reset_states()

    predictions = model.predict(test_gen, verbose=0)
    if batch_size > 1:
        predictions = predictions.reshape(-1, mini_batch_size)
        preds = []
        for j in range(batch_size):
            pred = np.concatenate((predictions[0+j], predictions[batch_size+j]), axis=0)
            for i in range(2, predictions.shape[0]//batch_size):
                pred = np.concatenate((pred, predictions[i*batch_size+j]), axis=0)
            preds.append(pred)

        predictions = np.array(preds, dtype=np.float32)
        y_test = test_gen.y[:, w:len(predictions[0])+w]
        x_test = test_gen.x[:, w:len(predictions[0])+w]

        y_test = np.array(y_test.reshape(-1), dtype=np.float32)
        x_test = np.array(x_test.reshape(-1), dtype=np.float32)
        predictions = predictions.reshape(-1)

    else:
        x_test, y_test = test_gen.getXY()
        predictions = predictions.reshape(-1)
        y_test = y_test.reshape(-1)
        x_test = x_test.reshape(-1)

        predictions = np.array(predictions, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32)
        x_test = np.array(x_test, dtype=np.float32)

    predictWaves(predictions, x_test, y_test, model_save_dir, save_folder, fs, '0')

    mse = tf.keras.metrics.mean_squared_error(y_test, predictions)
    mae = tf.keras.metrics.mean_absolute_error(y_test, predictions)
    esr = ESR(y_test, predictions)
    rmse = NRMSE(y_test, predictions)
    stft = STFT()(y_test, predictions)
    sf = SF()(y_test, predictions)

    results_ = {'mse': mse, 'mae': mae, 'esr': esr, 'rmse': rmse, 'stft': stft, 'sf': sf}

    with open(os.path.normpath('/'.join([model_save_dir, save_folder, str(model_name) + 'results.txt'])), 'w') as f:
        for key, value in results_.items():
            print('\n', key, '  : ', value, file=f)

    return 42