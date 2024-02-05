import os
import tensorflow as tf
from UtilsForTrainings import plotTraining, writeResults, checkpoints, predictWaves
from Models import create_model_ED, create_model_LSTM, create_model_LRU, create_model_S4D, MyLRScheduler
from DatasetsClass import DataGeneratorPickles
import numpy as np
import random
from Metrics import ESR, STFT_f, STFT_t, RMSE, flux, MFCC

def train(**kwargs):
    b_size = kwargs.get('b_size', 1)
    learning_rate = kwargs.get('learning_rate', 1e-4)
    units = kwargs.get('units', 16)
    model_save_dir = kwargs.get('model_save_dir', '../../TrainedModels/')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    inference = kwargs.get('inference', False)
    dataset = kwargs.get('dataset', 'FilterNeutron')
    model_name = kwargs.get('model', None)
    data_dir = kwargs.get('data_dir', '../../Files/')
    epochs = kwargs.get('epochs', [1, 60])

    epochs0 = epochs[0]
    epochs1 = epochs[1]

    #####seed
    np.random.seed(422)
    tf.random.set_seed(422)
    random.seed(422)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Defining the losses

    #lossesName = ['OutLayer', 'OutLayer_1']
    #losses = {
    #    lossesName[0]: 'mse',
    #    lossesName[1]: STFT,
    #}
    #lossWeights = {lossesName[0]: 1., lossesName[1]: 0.0000000001}#, lossesName[3]: w[3]}0.000001,

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
    elif dataset == 'BBD':
        data_dir = data_dir + 'BBD'
        D = 0
    elif dataset == 'ODNeutron':
        data_dir = data_dir + 'ODNeutron'
        D = 2
    elif dataset == 'Pultec':
        data_dir = data_dir + 'Pultec'
        D = 2
    elif dataset == 'SpringReverb':
        fs = 16000
        data_dir = data_dir + 'SpringReverb'
        D = 0
    elif dataset == 'LeslieWooferTR':
        data_dir = data_dir + 'LeslieWooferTR'
        fs = 16000
        D = 0
    else:
        data_dir = None
    e = 32
    d = 32
    w = e+d
    if model_name == 'LRU':
        model = create_model_LRU(cond_dim=D, input_dim=w, units=units+4, b_size=b_size)
    elif model_name == 'S4D':
        model = create_model_S4D(cond_dim=D, input_dim=w, units=units+4, b_size=b_size)
    elif model_name == 'LSTM':
        model = create_model_LSTM(cond_dim=D, input_dim=w, units=units, b_size=b_size)
    elif model_name == 'ED':
        model = create_model_ED(cond_dim=D, input_dim1=e, input_dim2=d, units=units, b_size=b_size)
    else:
        model = None
        
    metrics = ['mse', 'mae']

    callbacks = []
    ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(model_save_dir, save_folder)

    if not inference:
        callbacks += [ckpt_callback, ckpt_callback_latest]
        best = tf.train.latest_checkpoint(ckpt_dir)
        if best is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(best)
            # start_epoch = int(latest.split('-')[-1].split('.')[0])
            # print('Starting from epoch: ', start_epoch + 1)
        else:
            print("Initializing random weights.")

        # load the datasets
        train_gen = DataGeneratorPickles(data_dir, dataset + '_train.pickle', input_enc_size=e,
                                          input_dec_size=d, cond_size=D, model=model_name, batch_size=b_size)
        val_gen = DataGeneratorPickles(data_dir, dataset + '_val.pickle', input_enc_size=e,
                                        input_dec_size=d, cond_size=D, model=model_name, batch_size=b_size)
        
        loss_training = []
        loss_val = []
        training_steps = train_gen.training_steps
        opt = tf.keras.optimizers.Adam(learning_rate=MyLRScheduler(learning_rate, training_steps, epochs0, 4), clipnorm=1)
        model.compile(loss='mse', metrics=metrics, optimizer=opt)
        
        # train the model
        for i in range(epochs0, epochs1, 1):
            print(model.optimizer.learning_rate)

            results = model.fit(train_gen, epochs=1, verbose=0, shuffle=False, validation_data=val_gen, callbacks=callbacks)
            model.reset_states()

            loss_training.append(results.history['loss'])
            loss_val.append(results.history['val_loss'])
            print('epochs:', i+1)
        # save results
        writeResults(results, units, epochs, b_size, learning_rate, model_save_dir,
                     save_folder, epochs0)

        loss_training.append(results.history['loss'])
        loss_val.append(results.history['val_loss'])
        plotTraining(loss_training, loss_val, model_save_dir, save_folder)

        print("Training done")

    best = tf.train.latest_checkpoint(ckpt_dir)
    if best is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(best).expect_partial()

    # compute test loss
    test_gen = DataGeneratorPickles(data_dir, dataset + '_test.pickle', input_enc_size=e, input_dec_size=d,
                                    cond_size=D, model=model_name, batch_size=b_size)

    model.reset_states()
    predictions = model.predict(test_gen, verbose=0)[:, 0]
    predictWaves(predictions, test_gen.x[w:len(predictions)+w], test_gen.y[w:len(predictions)+w], model_save_dir, save_folder, fs, '0')

    return 42