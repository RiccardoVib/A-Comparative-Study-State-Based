import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from scipy.io import wavfile
#from scipy import fft
from Utils import filterAudio

def writeResults(results, units, epochs, b_size, learning_rate, model_save_dir,
                 save_folder,
                 index):
    results = {
        'Min_val_loss': np.min(results.history['val_loss']),
        'Min_train_loss': np.min(results.history['loss']),
        'b_size': b_size,
        'learning_rate': learning_rate,
        # 'Train_loss': results.history['loss'],
        'Val_loss': results.history['val_loss'],
        'units': units,
        'epochs': epochs
    }
    with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results_' + str(index) + '.txt'])), 'w') as f:
        for key, value in results.items():
            print('\n', key, '  : ', value, file=f)
        pickle.dump(results,
                    open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results_' + str(index) + '.pkl'])),
                         'wb'))


def plotResult(predictions, x, y, model_save_dir, save_folder, fs, filename):
    l = len(x) // (fs*5)
    pred = predictions
    tar = y
    for i in range(0, l-1):
        y = tar[i * fs*5: (i + 1) * fs*5]
        predictions = pred[i * fs*5: (i + 1) * fs*5]
        inp = x[i * fs*5: (i + 1) * fs*5]
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(inp, label='input', alpha=0.3)
        ax.plot(y, label='Target', alpha=0.9)
        ax.plot(predictions, label='Prediction', alpha=0.7)
        # ax.label_outer()
        ax.legend(loc='upper right')
        fig.savefig(model_save_dir + '/' + save_folder + '/plot' + filename + str(i) + '.pdf', format='pdf')
        plt.close('all')
        #N_fft = 2048
        #N_stft = 2048
        # FFT
#         FFT_t = np.abs(fft.fftshift(fft.fft(y, n=N_fft))[N_fft // 2:])
#         FFT_p = np.abs(fft.fftshift(fft.fft(predictions, n=N_fft))[N_fft // 2:])
#         freqs = fft.fftshift(fft.fftfreq(N_fft) * fs)
#         freqs = freqs[N_fft // 2:]
#
#         fig, ax = plt.subplots(1, 1)
#         ax.semilogx(freqs, 20 * np.log10(np.abs(FFT_t)), label='Target', )
#         ax.semilogx(freqs, 20 * np.log10(np.abs(FFT_p)), label='Prediction')
#         ax.set_xlabel('Frequency')
#         ax.set_ylabel('Magnitude (dB)')
#         ax.axis(xmin=20, xmax=22050)
#
#         ax.legend(loc='upper right')
#         fig.savefig(model_save_dir + '/' + save_folder + '/FFT' + str(i) + '.pdf', format='pdf')
#         plt.close('all')
#
#         # STFT
#         D = librosa.stft(y, n_fft=N_stft)
#         S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
#         fig, ax = plt.subplots(nrows=2, ncols=1)
#         librosa.display.specshow(S_db, sr=fs, y_axis='linear', x_axis='time', ax=ax[0])
#         ax[0].set_title('STFT Magnitude (Top: target, Bottom: prediction)')
#         ax[0].set_ylabel('Frequency [Hz]')
#         ax[0].set_xlabel('Time [sec]')
#         ax[0].label_outer()
#
#         D = librosa.stft(predictions, n_fft=N_stft)
#         S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
#         librosa.display.specshow(S_db, sr=fs, y_axis='linear', x_axis='time', ax=ax[1])
#         ax[1].set_ylabel('Frequency [Hz]')
#         ax[1].set_xlabel('Time [sec]')
#         fig.savefig(model_save_dir + '/' + save_folder + '/STFT' + str(i) + '.pdf', format='pdf')
#         plt.close('all')
#
#
def plotTraining(loss_training, loss_val, model_save_dir, save_folder):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(np.array(loss_training), label='train'),
    ax.plot(np.array(loss_val), label='validation')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('train vs. validation accuracy')    
    plt.legend(loc='upper center')  # , bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2)
    fig.savefig(model_save_dir + '/' + save_folder + '/' + 'loss.png')
    plt.close('all')


def predictWaves(predictions, x_test, y_test, model_save_dir, save_folder, fs, filename):
    pred_name = filename + '_pred.wav'
    inp_name = filename + '_inp.wav'
    tar_name = filename + '_tar.wav'

    pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
    inp_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', inp_name))
    tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

    if not os.path.exists(os.path.dirname(pred_dir)):
        os.makedirs(os.path.dirname(pred_dir))

    predictions = filterAudio(predictions.reshape(-1))
    predictions = (predictions.reshape(-1))    
    x_test = x_test.reshape(-1)
    y_test = y_test.reshape(-1)

    wavfile.write(pred_dir, fs, predictions.reshape(-1))
    wavfile.write(inp_dir, fs, x_test.reshape(-1))
    wavfile.write(tar_dir, fs, y_test.reshape(-1))

    plotResult(predictions, x_test, y_test, model_save_dir, save_folder, fs, filename)


def checkpoints(model_save_dir, save_folder):
    ckpt_path = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'best', 'best.ckpt'))
    ckpt_path_latest = os.path.normpath(
        os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest', 'latest.ckpt'))
    ckpt_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'best'))
    ckpt_dir_latest = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest'))

    if not os.path.exists(os.path.dirname(ckpt_dir)):
        os.makedirs(os.path.dirname(ckpt_dir))
    if not os.path.exists(os.path.dirname(ckpt_dir_latest)):
        os.makedirs(os.path.dirname(ckpt_dir_latest))

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, monitor='val_loss', mode='min',
                                                       save_best_only=True, save_weights_only=True, verbose=1)
    ckpt_callback_latest = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path_latest, monitor='val_loss',
                                                              mode='min',
                                                              save_best_only=False, save_weights_only=True,
                                                              verbose=1)

    return ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest
