import pickle
import os
import numpy as np
from scipy.signal.windows import tukey
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from Utils import filterAudio


class DataGeneratorPickles(Sequence):

    def __init__(self, data_dir, filename, input_enc_size, input_dec_size, cond_size, model, batch_size=10):
        """
        Initializes a data generator object
          :param data_dir: the directory in which data are stored
          :param output_size: output size
          :param batch_size: The size of each batch returned by __getitem__
        """
        file_data = open(os.path.normpath('/'.join([data_dir, filename])), 'rb')
        Z = pickle.load(file_data)

        self.x = np.array(Z['x'][:, :], dtype=np.float32)
        self.y = np.array(Z['y'][:, :], dtype=np.float32)
        self.batch_size = batch_size

        rep = self.x.shape[1]
        self.x = self.x.reshape(-1)
        self.y = self.y.reshape(-1)
        lim = int((self.x.shape[0] / self.batch_size) * self.batch_size)
        self.x = self.x[:lim]
        self.y = self.y[:lim]

        if cond_size != 0:
            self.z = np.array(Z['z'][:, :], dtype=np.float32)
            self.z = np.repeat(self.z, rep, axis=1)
        del Z
        self.input_enc_size = input_enc_size
        self.input_dec_size = input_dec_size
        self.window = int(self.input_enc_size + self.input_dec_size)
        self.training_steps = (lim // self.batch_size)

        self.cond_size = cond_size
        self.model = model
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(self.x.shape[0] + self.window - 1)

    def __len__(self):
        return int((self.x.shape[0]) / self.batch_size) - 1

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):
        ## Initializing Batch
        X = []  # np.empty((self.batch_size, 2*self.w))
        Y = []  # np.empty((self.batch_size, self.output_size))
        Z = []  # np.empty((self.batch_size, self.cond_size))

        # get the indices of the requested batch
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size] + self.window

        if self.cond_size != 0:
            for t in range(indices[0], indices[-1] + 1, 1):
                X.append(np.array(self.x[t - self.window: t]).T)
                Y.append(np.array(self.y[t - 1]).T)
                Z.append(np.array(self.z[:, t - 1]).T)

            X = np.array(X, dtype=np.float32)
            Y = np.array(Y, dtype=np.float32)
            Z = np.array(Z, dtype=np.float32)
        else:
            for t in range(indices[0], indices[-1] + 1, 1):
                X.append(np.array(self.x[t - self.window: t]).T)
                Y.append(np.array(self.y[t - 1]).T)
            X = np.array(X, dtype=np.float32)
            Y = np.array(Y, dtype=np.float32)

        if self.model == 'LRU' or self.model == 'LSTM' or self.model == 'LSTM2' or self.model == 'S4D':
            if self.cond_size != 0:
                return [Z, X], Y
            else:
                return X, Y
        else:
            if self.cond_size != 0:
                return [Z, X[:, :self.input_enc_size], X[:, -self.input_dec_size:]], Y
            else:
                return [X[:, :self.input_enc_size], X[:, -self.input_dec_size:]], Y
