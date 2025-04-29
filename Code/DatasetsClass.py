import pickle
import os
import numpy as np
from tensorflow.keras.utils import Sequence
from scipy.signal.windows import tukey
#import matplotlib.pyplot as plt


class DataGeneratorPickles(Sequence):

    def __init__(self, data_dir, filename, input_size, cond, mini_batch_size=1, batch_size=9, set='train', model=None,
                 stateful=False):
        """
        Initializes a data generator object
          :param data_dir: the directory in which data are stored
          :param filename: dataset's filename
          :param batch_size: The size of each batch returned by __getitem__
        """

        self.indices2 = None
        self.indices = None
        self.data_dir = data_dir
        self.filename = filename
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.window = input_size
        self.model = model
        self.cond = cond
        self.stateful = stateful
        self.set = set

        self.x, self.y, self.z = self.prepareXYZ(data_dir, filename)
        assert self.x.shape[0] % self.batch_size == 0

        self.idj = 0
        self.idx = -1

        self.max_1 = ((self.x.shape[1]) // self.mini_batch_size)-1
        self.max_2 = (self.x.shape[0] // self.batch_size)
        self.max = self.max_1 * self.max_2
        self.training_steps = self.max
        self.on_epoch_end()

    def prepareXYZ(self, data_dir, filename):
        file_data = open(os.path.normpath('/'.join([data_dir, filename])), 'rb')
        Z = pickle.load(file_data)
        x = np.array(Z['x'][:, :], dtype=np.float32)
        y = np.array(Z['y'][:, :], dtype=np.float32)

        x = x * np.array(tukey(x.shape[1], alpha=0.000005), dtype=np.float32).reshape(1, -1)
        y = y * np.array(tukey(x.shape[1], alpha=0.000005), dtype=np.float32).reshape(1, -1)

        if x.shape[0] == 1:
            x = np.repeat(x, y.shape[0], axis=0)

        z = np.array(Z['z'], dtype=np.float32)
        del Z

        N = int((x.shape[1] - self.window) / self.mini_batch_size)  # how many iteration
        lim = int(N * self.mini_batch_size)  # how many samples
        x = x[:, :lim]
        y = y[:, :lim]

        rep = x.shape[1]

        if z.shape[0] < z.shape[1]:
            z = z.T
        z = np.repeat(z[:, np.newaxis, :], rep, axis=1)

        return x, y, z

    def on_epoch_end(self):

        self.indices = np.arange(self.window, self.x.shape[1])
        self.indices2 = np.arange(0, self.x.shape[0])
        self.idj = 0
        self.idx = -1
        if self.stateful:
            self.model.layers[4].reset_states()

    def __len__(self):
        return int(self.max)

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def getXY(self):

        Xs, Ys = [], []
        for i in range(self.__len__()):

            ## Initializing Batch
            X = np.zeros((self.batch_size, self.mini_batch_size, 1))
            Y = np.zeros((self.batch_size, self.mini_batch_size, 1))
            Z = np.zeros((self.batch_size, self.mini_batch_size, self.cond))

            if i == 0:
                self.idj = 0
                self.idx = -1

            if i % self.max_1 - 1 == 0 and i != 1:
                self.idj += 1
                self.idx = -1
                if self.stateful:
                    self.model.layers[4].reset_states()

            self.idx += 1

            # get the indices of the requested batch
            indices = self.indices[self.idx * self.mini_batch_size:(self.idx + 1) * self.mini_batch_size]
            indices2 = self.indices2[self.idj * self.batch_size:(self.idj + 1) * self.batch_size]
            c = 0

            for t in range(indices[0], indices[-1] + 1, 1):
                X[:, c, :] = np.array(self.x[indices2, t - 1: t])
                Y[:, c, :] = np.array(self.y[indices2, t - 1:t])
                Z[:, c, :] = np.array(self.z[indices2, t - 1])
                c += 1

            Xs.append(X)
            Ys.append(Y)

        return np.array(Xs), np.array(Ys)

    def __getitem__(self, idx):
        ## Initializing Batch
        X = np.zeros((self.batch_size, self.mini_batch_size, self.window))
        Y = np.zeros((self.batch_size, self.mini_batch_size, 1))
        Z = np.zeros((self.batch_size, self.mini_batch_size, self.cond))

        if idx == 0:
            self.idj = 0
            self.idx = -1

        if idx % self.max_1 - 1 == 0 and idx != 1:
            self.idj += 1
            self.idx = -1
            if self.stateful:
                self.model.layers[4].reset_states()

        self.idx += 1

        # get the indices of the requested batch
        indices = self.indices[self.idx * self.mini_batch_size:(self.idx + 1) * self.mini_batch_size]
        indices2 = self.indices2[self.idj * self.batch_size:(self.idj + 1) * self.batch_size]
        c = 0

        for t in range(indices[0], indices[-1] + 1, 1):
            X[:, c, :] = np.array(self.x[indices2, t - self.window: t])
            Y[:, c, :] = np.array(self.y[indices2, t - 1:t])
            Z[:, c, :] = np.array(self.z[indices2, t - 1])
            c += 1

        return [Z, X], Y