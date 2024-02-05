import librosa
import tensorflow as tf
from scipy import signal
import numpy as np
from tensorflow.keras import backend as K

def ESR(y_true, y_pred):
    return tf.divide(K.mean(K.square(y_pred - y_true)), K.mean(K.square(y_true) + 0.00001))

def RMSE(y_true, y_pred):
    return K.mean(K.abs(K.sqrt(K.square(K.abs(y_pred))) - K.sqrt(K.square(K.abs(y_true)))))

def flux(y_true, y_pred, sr):

    y_t = []
    y_p = []
    w = 480000
    for i in range(0, len(y_true)-w, w//4):
        y_t.append(librosa.onset.onset_strength(y=y_true[i:i+w], sr=sr))
        y_p.append(librosa.onset.onset_strength(y=y_pred[i:i+w], sr=sr))
    y_t = np.array(y_t)
    y_p = np.array(y_p)
    y_true = y_t/y_t.max()
    y_pred = y_p/y_p.max()

    return K.mean(tf.abs(y_true - y_pred))

def First_difference(y_true, y_pred):
    begin_back = [0 for _ in range(3)]
    begin_front = [0 for _ in range(3)]
    begin_front[1] = 1

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    Y_true = K.abs(
        tf.signal.stft(y_true, fft_length=512, frame_length=512, frame_step=512 // 4, pad_end=True))
    Y_pred = K.abs(
        tf.signal.stft(y_pred, fft_length=512, frame_length=512, frame_step=512 // 4, pad_end=True))

    shape = Y_true.shape
    shape[1] -= 1

    slice_front = tf.slice(Y_true, begin_front, shape)
    slice_back = tf.slice(Y_pred, begin_back, shape)
    d_t = slice_front - slice_back

    slice_front = tf.slice(Y_pred, begin_front, shape)
    slice_back = tf.slice(Y_pred, begin_back, shape)
    d_p = slice_front - slice_back

    return tf.norm((d_t - d_p), ord=1)


def MFCC(y_true, y_pred, sr):
    y_true_ = tf.reshape(y_true, [-1])
    y_pred_ = tf.reshape(y_pred, [-1])
    m = [1024]
    loss = 0
    for i in range(len(m)):

        pad_amount = int(m[i] // 2)  # Symmetric even padding.
        pads = [[pad_amount, pad_amount]]
        y_true = tf.pad(y_true_, pads, mode='CONSTANT', constant_values=0)
        y_pred = tf.pad(y_pred_, pads, mode='CONSTANT', constant_values=0)

        Y_true = tf.signal.stft(y_true, frame_length=m[i], frame_step=m[i]//4,  fft_length=m[i], pad_end=False)
        Y_pred = tf.signal.stft(y_pred, frame_length=m[i], frame_step=m[i]//4,  fft_length=m[i], pad_end=False)

        Y_true = tf.abs(Y_true)
        Y_pred = tf.abs(Y_pred)

        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = Y_true.shape[-1]
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 30.0, sr//2, 80
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sr, lower_edge_hertz,
        upper_edge_hertz)

        mel_spectrograms_pred = tf.tensordot(Y_pred, linear_to_mel_weight_matrix, 1)
        mel_spectrograms_pred.set_shape(Y_pred.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

        mel_spectrograms_tar = tf.tensordot(Y_true, linear_to_mel_weight_matrix, 1)
        mel_spectrograms_tar.set_shape(Y_true.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))


        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms_pred = tf.math.log(mel_spectrograms_pred + 1e-6)

        # Compute MFCCs from log_mel_spectrograms
        mfccs_pred = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms_pred)

        log_mel_spectrograms_tar = tf.math.log(mel_spectrograms_tar + 1e-6)

        # Compute MFCCs from log_mel_spectrograms
        mfccs_tar = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms_tar)

        loss += tf.norm((mfccs_tar - mfccs_pred), ord=1) / (tf.norm(mfccs_tar, ord=1))

    return loss

    
def STFT_t(y_true, y_pred):#auraloss multi-STFT
    m = [32, 64, 128]
    y_true_ = tf.reshape(y_true, [-1])
    y_pred_ = tf.reshape(y_pred, [-1])

    loss = 0
    log_loss = 0
    for i in range(len(m)):

        pad_amount = int(m[i] // 2)  # Symmetric even padding like librosa.
        pads = [[pad_amount, pad_amount]]
        y_true = tf.pad(y_true_, pads, mode='CONSTANT', constant_values=0)
        y_pred = tf.pad(y_pred_, pads, mode='CONSTANT', constant_values=0)

        Y_true = K.abs(tf.signal.stft(y_true, fft_length=m[i], frame_length=m[i], frame_step=m[i] // 4, pad_end=False))
        Y_pred = K.abs(tf.signal.stft(y_pred, fft_length=m[i], frame_length=m[i], frame_step=m[i] // 4, pad_end=False))

        Y_true = tf.pow(K.abs(Y_true), 2)
        Y_pred = tf.pow(K.abs(Y_pred), 2)

        loss += tf.norm((Y_true - Y_pred), ord=1) / (tf.norm(Y_true, ord=1) + 0.00001)

    return (loss + log_loss) / len(m)

def STFT_f(y_true, y_pred):#auraloss multi-STFT
    m = [256, 512, 1024]
    y_true_ = tf.reshape(y_true, [-1])
    y_pred_ = tf.reshape(y_pred, [-1])

    loss = 0
    log_loss = 0
    for i in range(len(m)):

        pad_amount = int(m[i] // 2)  # Symmetric even padding like librosa.
        pads = [[pad_amount, pad_amount]]
        y_true = tf.pad(y_true_, pads, mode='CONSTANT', constant_values=0)
        y_pred = tf.pad(y_pred_, pads, mode='CONSTANT', constant_values=0)

        Y_true = K.abs(tf.signal.stft(y_true, fft_length=m[i], frame_length=m[i], frame_step=m[i] // 4, pad_end=False))
        Y_pred = K.abs(tf.signal.stft(y_pred, fft_length=m[i], frame_length=m[i], frame_step=m[i] // 4, pad_end=False))

        Y_true = tf.pow(K.abs(Y_true), 2)
        Y_pred = tf.pow(K.abs(Y_pred), 2)

        loss += (tf.norm((Y_true - Y_pred), ord=1) / (tf.norm(Y_true, ord=1) + 0.00001))

    return (loss + log_loss) / len(m)

def SF(y_true, y_pred): #Spectral difference
    window = 512

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    Y_true = K.abs(tf.signal.stft(y_true, fft_length=512, frame_length=window, frame_step=window, pad_end=True))
    Y_pred = K.abs(tf.signal.stft(y_pred, fft_length=512, frame_length=window, frame_step=window, pad_end=True))

    loss = 0.
    for n in range(1, tf.shape(Y_true)[0]):
        sf_true = tf.reduce_sum(tf.divide(K.abs(Y_true[n]) - K.abs(Y_true[n - 1] + K.abs(K.abs(Y_true[n]) - K.abs(Y_true[n - 1]))), 2) ** 2)
        sf_pred = tf.reduce_sum(tf.divide(K.abs(Y_pred[n]) - K.abs(Y_pred[n - 1] + K.abs(K.abs(Y_pred[n]) - K.abs(Y_pred[n - 1]))), 2) ** 2)
        sf_true = tf.cast(sf_true, tf.float32)
        sf_pred = tf.cast(sf_pred, tf.float32)
        loss += (sf_true - sf_pred)
    return tf.reduce_mean(loss)

