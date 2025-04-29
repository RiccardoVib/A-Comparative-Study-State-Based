import tensorflow as tf
from tensorflow.keras import backend as K

def diff(x, axis=-1):
  """Take the finite difference of a tensor along an axis.

  Args:
    x: Input tensor of any dimension.
    axis: Axis on which to take the finite difference.

  Returns:
    d: Tensor with size less than x by 1 along the difference dimension.

  Raises:
    ValueError: Axis out of range for tensor.
  """
  shape = x.shape.as_list()
  ndim = len(shape)
  if axis >= ndim:
    raise ValueError('Invalid axis index: %d for tensor with only %d axes.' %
                     (axis, ndim))

  begin_back = [0 for _ in range(ndim)]
  begin_front = [0 for _ in range(ndim)]
  begin_front[axis] = 1

  shape[axis] -= 1
  slice_front = tf.slice(x, begin_front, shape)
  slice_back = tf.slice(x, begin_back, shape)
  d = slice_front - slice_back
  return d

def NRMSE(y_true, y_pred):
    return tf.divide(K.mean(K.abs(K.sqrt(K.square(K.abs(y_pred))) - K.sqrt(K.square(K.abs(y_true))))),  K.mean(K.sqrt(K.square(K.abs(y_true))) + 0.00001))


def ESR(y_true, y_pred):#auraloss
    return tf.divide(K.mean(K.square(y_pred - y_true)), K.mean(K.square(y_true) + 0.00001))


class SF(tf.keras.losses.Loss):
    def __init__(self, m=[2048], name="STFT_f", **kwargs):
        super().__init__(name=name, **kwargs)
        self.m = m
        self.delta = 1e-12

    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [1, -1])
        y_pred = tf.reshape(y_pred, [1, -1])

        loss = 0.
        for i in range(len(self.m)):


            Y_true = K.abs(
                tf.signal.stft(y_true, fft_length=self.m[i], frame_length=self.m[i], frame_step=self.m[i] // 4,
                               pad_end=False))
            Y_pred = K.abs(
                tf.signal.stft(y_pred, fft_length=self.m[i], frame_length=self.m[i], frame_step=self.m[i] // 4,
                               pad_end=False))

            Y_true = diff(Y_true, axis=2)
            Y_pred = diff(Y_pred, axis=2)

            #Y_true = tf.cast(tf.pow((Y_true), 2), dtype=tf.float32)
            #Y_pred = tf.cast(tf.pow((Y_pred), 2), dtype=tf.float32)

            loss += tf.reduce_mean(tf.divide(tf.norm((Y_true - Y_pred), ord=1), tf.norm(Y_true, ord=1)))

        return tf.get_static_value(loss/len(self.m))

    def get_config(self):
        config = {
            'm': self.m
        }
        base_config = super().get_config()
        return {**base_config, **config}

class STFT(tf.keras.losses.Loss):
    def __init__(self, m=[256, 512, 1024], name="STFT", **kwargs):
        super().__init__(name=name, **kwargs)
        self.m = m
        self.delta = 1e-12

    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [1, -1])
        y_pred = tf.reshape(y_pred, [1, -1])

        loss = 0.
        loss_l = 0.
        for i in range(len(self.m)):


            Y_true = K.abs(
                tf.signal.stft(y_true, fft_length=self.m[i], frame_length=self.m[i], frame_step=self.m[i] // 4,
                               pad_end=False))
            Y_pred = K.abs(
                tf.signal.stft(y_pred, fft_length=self.m[i], frame_length=self.m[i], frame_step=self.m[i] // 4,
                               pad_end=False))

            Y_true = tf.cast(tf.pow((Y_true), 2), dtype=tf.float32)
            Y_pred = tf.cast(tf.pow((Y_pred), 2), dtype=tf.float32)

            l_true = K.log(Y_true + self.delta)
            l_pred = K.log(Y_pred + self.delta)

            loss += tf.reduce_mean(tf.divide(tf.norm((Y_true - Y_pred), ord=1), tf.norm(Y_true, ord=1)))
            loss_l += tf.reduce_mean(tf.divide(tf.norm((l_true - l_pred), ord=1), tf.norm(l_true, ord=1)))

        return tf.get_static_value(loss/len(self.m)) + tf.get_static_value(loss_l/len(self.m))

    def get_config(self):
        config = {
            'm': self.m
        }
        base_config = super().get_config()
        return {**base_config, **config}