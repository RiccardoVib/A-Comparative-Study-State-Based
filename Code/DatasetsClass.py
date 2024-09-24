import pickle
import os
import numpy as np
from tensorflow.keras.utils import Sequence
from scipy.signal.windows import tukey


class DataGeneratorPickles(Sequence):

    def __init__(self, data_dir, filename, input_size, cond, batch_size=10):
        """
        Initializes a data generator object for the CL1B dataset
          :param filename: the name of the dataset
          :param data_dir: the directory in which data are stored
          :param input_size: the input size
          :param cond: the number of coniditoning parameters
          :param batch_size: The size of each batch returned by __getitem__
        """
        self.cond = cond
        self.data_dir = data_dir
        self.filename = filename
        self.batch_size = batch_size
        self.window = input_size

        # prepare the input, target and conditioning matrix
        self.x, self.y, self.z, rep, lim = self.prepareXYZ(data_dir, filename)
        self.training_steps = (lim // self.batch_size)
        self.on_epoch_end()

    def prepareXYZ(self, data_dir, filename):
        # load all the audio files
        file_data = open(os.path.normpath('/'.join([data_dir, filename])), 'rb')
        Z = pickle.load(file_data)
        x = np.array(Z['x'][:, :], dtype=np.float32)
        y = np.array(Z['y'][:, :], dtype=np.float32)

        # if input is shared to all the targets, it is repeat accordingly to the number of target audio files
        if x.shape[0] == 1:
            x = np.repeat(x, y.shape[0], axis=0)

        # windowing the signal to avoid misalignments
        x = x * np.array(tukey(x.shape[1], alpha=0.000005), dtype=np.float32).reshape(1, -1)
        y = y * np.array(tukey(x.shape[1], alpha=0.000005), dtype=np.float32).reshape(1, -1)

        z = np.array(Z['z'], dtype=np.float32).T
        if z.shape[0] < z.shape[1]:
            z = z.T
        del Z

        # reshape to one dimension
        rep = x.shape[1]
        x = x.reshape(-1)
        y = y.reshape(-1)

        # how many iterations are needed
        N = int((x.shape[0] - self.window) / self.batch_size) - 1

        # remove the last samples if not enough for a batch
        lim = int(N * self.batch_size) + self.window
        x = x[:lim]
        y = y[:lim]
        z = np.repeat(z, rep, axis=0)

        return x, y, z, rep, lim

    def on_epoch_end(self):
        # create/reset the vector containing the indices of the batches
        self.indices = np.arange(self.window, self.x.shape[0]+1)
        self.count = 0


    def __len__(self):
        # compute the needed number of iteration before conclude one epoch
        return int((self.x.shape[0]) / self.batch_size)

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):
        # Initializing input, target, and conditioning batches
        X = np.empty((self.batch_size, self.window))
        Y = np.empty((self.batch_size, 1))
        Z = np.empty((self.batch_size, self.cond))

        # get the indices of the requested batch
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        c = 0
        for t in range(indices[0], indices[-1]+1, 1):
            X[c, :] = np.array(self.x[t - self.window: t])
            Y[c, :] = np.array(self.y[t-1])
            Z[c, :] = np.array(self.z[t-1])
            c = c + 1

        return [Z, X], Y

