import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Conv1D, LSTM, Multiply, ReLU, BatchNormalization, PReLU

from tensorflow.keras.models import Model
from Layers import LRU, S4D, GLU


class MyLRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, training_steps, epochs, order):
        self.initial_learning_rate = (initial_learning_rate)
        self.steps = training_steps*epochs
        self.order = order
    def __call__(self, step):
        return tf.cast(self.initial_learning_rate / ( tf.cast(step/self.steps, dtype=tf.float32) + 1.)**self.order, dtype=tf.float32)

def create_model_S4D(cond_dim, input_dim, units, b_size=2399, drop=0.):
    T = input_dim
    D = cond_dim

    # Defining decoder inputs
    decoder_inputs = tf.keras.layers.Input(batch_shape=(b_size, T), name='dec_input')
    decoder_outputs_ = tf.keras.layers.Dense(units // 2, input_shape=(b_size, T), name='LinearProjection')(
        decoder_inputs)
    decoder_outputs = tf.reshape(decoder_outputs_, [b_size, units // 2])

    # Defining encoder inputs
    if D != 0:
        cond_inputs = tf.keras.layers.Input(batch_shape=(b_size, D), name='cond')

    decoder_outputs = S4D(units)(decoder_outputs)
    decoder_outputs = tf.keras.layers.Dense(units // 2, activation='softsign', name='NonlinearDenseLayer')(
        decoder_outputs)
    decoder_outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(decoder_outputs)#+decoder_outputs_

    if D != 0:
        film = tf.keras.layers.Dense(units, batch_input_shape=(b_size, D))(cond_inputs)
        g, b = tf.split(film, 2, axis=-1)
        decoder_outputs = tf.keras.layers.Multiply()([decoder_outputs, g])
        decoder_outputs = tf.keras.layers.Add()([decoder_outputs, b])
        decoder_outputs = GLU(units//2)(decoder_outputs)

        decoder_outputs = tf.keras.layers.Dense(1, name='OutLayer')(decoder_outputs)
        decoder_outputs = tf.keras.layers.Add()([decoder_outputs, decoder_inputs[:, -1]])
        model = tf.keras.models.Model([cond_inputs, decoder_inputs], decoder_outputs)
    else:
        decoder_outputs = tf.keras.layers.Dense(1, name='OutLayer')(decoder_outputs)
        decoder_outputs = tf.keras.layers.Add()([decoder_outputs, decoder_inputs[:, -1]])

        model = tf.keras.models.Model(decoder_inputs, decoder_outputs)
    model.summary()
    return model


def create_model_LRU(cond_dim, input_dim, units, b_size=2399):
    T = input_dim
    D = cond_dim

    # Defining decoder inputs
    decoder_inputs = tf.keras.layers.Input(batch_shape=(b_size, T), name='dec_input')
    decoder_outputs_ = tf.keras.layers.Dense(units // 2, input_shape=(b_size, T), name='LinearProjection')(
        decoder_inputs)
    decoder_outputs = tf.reshape(decoder_outputs_, [b_size, units // 2])

    # Defining encoder inputs
    if D != 0:
        cond_inputs = tf.keras.layers.Input(batch_shape=(b_size, D), name='cond')

    decoder_outputs = LRU(N=units, H=units // 2)(decoder_outputs)  # units//4
    decoder_outputs = tf.keras.layers.Dense(units // 2, activation='softsign', name='NonlinearDenseLayer')(
        decoder_outputs)
    decoder_outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(decoder_outputs)#+decoder_outputs_

    if D != 0:
        film = tf.keras.layers.Dense(units, batch_input_shape=(b_size, D))(cond_inputs)
        g, b = tf.split(film, 2, axis=-1)
        decoder_outputs = tf.keras.layers.Multiply()([decoder_outputs, g])
        decoder_outputs = tf.keras.layers.Add()([decoder_outputs, b])
        decoder_outputs = GLU(units//2)(decoder_outputs)

        decoder_outputs = tf.keras.layers.Dense(1, name='OutLayer')(decoder_outputs)
        decoder_outputs = tf.keras.layers.Add()([decoder_outputs, decoder_inputs[:, -1]])

        model = tf.keras.models.Model([cond_inputs, decoder_inputs], decoder_outputs)
    else:
        decoder_outputs = tf.keras.layers.Dense(1, name='OutLayer')(decoder_outputs)
        decoder_outputs = tf.keras.layers.Add()([decoder_outputs, decoder_inputs[:, -1]])

        model = tf.keras.models.Model(decoder_inputs, decoder_outputs)
    model.summary()
    return model



def create_model_ED(cond_dim, input_dim1, input_dim2, units, b_size=2399, drop=0.):
    T = input_dim1  # time window
    T2 = input_dim2
    D = cond_dim

    encoder_inputs = Input(batch_shape=(b_size, T, 1), name='encoder_input')
    encoder_outputs, h, c = LSTM(units, stateful=True, return_sequences=False, return_state=True, name='LSTM_encoder',
                                 dropout=drop)(encoder_inputs)

    decoder_inputs = Input(shape=(T2), name='decoder_input')
    decoder_outputs = tf.keras.layers.Dense(units//2, input_shape=(b_size, T2), name='LinearProjection')(decoder_inputs)
    decoder_outputs = tf.expand_dims(decoder_outputs, axis=-1)
    decoder_outputs = LSTM(units, return_sequences=False, return_state=False, name='LSTM_decoder', dropout=drop)(decoder_outputs, initial_state=[h, c])

    decoder_outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(decoder_outputs)

    if D != 0:
        cond_inputs = Input(shape=(D), name='conditioning_input')
        film = Dense(units*2, batch_input_shape=(b_size, units))(cond_inputs)
        g, b = tf.split(film, 2, axis=-1)
        decoder_outputs = Multiply()([decoder_outputs, g])
        decoder_outputs = Add()([decoder_outputs, b])
        decoder_outputs = GLU(units//2)(decoder_outputs)

        decoder_outputs = Dense(1, name='OutLayer')(decoder_outputs)
        decoder_outputs = tf.keras.layers.Add()([decoder_outputs, decoder_inputs[:, -1]])

        model = Model([cond_inputs, encoder_inputs, decoder_inputs], decoder_outputs)
    else:
        decoder_outputs = Dense(1, name='OutLayer')(decoder_outputs)
        decoder_outputs = tf.keras.layers.Add()([decoder_outputs, decoder_inputs[:, -1]])

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()

    return model



def create_model_LSTM(cond_dim, input_dim, units, b_size=2399, drop=0.):
    T = input_dim
    D = cond_dim


    # Defining decoder inputs
    decoder_inputs = Input(batch_shape=(b_size, T), name='dec_input')    
    decoder_outputs = tf.keras.layers.Dense(units//2, input_shape=(b_size, T), name='LinearProjection')(decoder_inputs)

    # Defining encoder inputs
    if D != 0:
        cond_inputs = Input(shape=(D), name='enc_cond')

    decoder_outputs = tf.expand_dims(decoder_outputs, axis=1)
        
    decoder_outputs = LSTM(units, stateful=True, return_sequences=False, return_state=False, dropout=drop, name='LSTM')(
        decoder_outputs)
    decoder_outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(decoder_outputs)#+decoder_outputs_

    if D != 0:
        film = Dense(units*2, batch_input_shape=(b_size, units))(cond_inputs)
        g, b = tf.split(film, 2, axis=-1)
        decoder_outputs = Multiply()([decoder_outputs, g])
        decoder_outputs = Add()([decoder_outputs, b])
        decoder_outputs = GLU(units//2)(decoder_outputs)

        decoder_outputs = Dense(1, name='OutLayer')(decoder_outputs)
        decoder_outputs = tf.keras.layers.Add()([decoder_outputs, decoder_inputs[:, -1]])

        model = Model([cond_inputs, decoder_inputs], decoder_outputs)
    else:
        decoder_outputs = Dense(1, name='OutLayer')(decoder_outputs)
        decoder_outputs = tf.keras.layers.Add()([decoder_outputs, decoder_inputs[:, -1]])

        model = Model(decoder_inputs, decoder_outputs)

    model.summary()
    return model
