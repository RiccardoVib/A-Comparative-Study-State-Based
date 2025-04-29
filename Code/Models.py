import tensorflow as tf
from Layers import FiLM, ED_sharing_state
from S4D import S4D
from LRU import LRU
from S6 import S6


def create_model_S6(cond_dim, input_dim, units, mini_batch_size, b_size, stateful):
    T = input_dim
    D = cond_dim

    # Defining decoder inputs
    inputs = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, T), name='dec_input')

    decoder_outputs = tf.keras.layers.Dense(units // 2, name='LinearProjection')(inputs)
    decoder_outputs = tf.reshape(decoder_outputs, [b_size, mini_batch_size, units // 2])

    # Defining encoder inputs
    cond_inputs = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, D), name='cond')

    decoder_outputs = S6(model_input_dims=units//2, model_states=units//2, batch_size=b_size, stateful=stateful)(decoder_outputs)

    decoder_outputs = tf.keras.layers.Dense(units // 2, activation='tanh', name='NonlinearDenseLayer')(
        decoder_outputs)

    decoder_outputs = FiLM(units // 2)(decoder_outputs, cond_inputs)

    decoder_outputs = tf.keras.layers.Dense(1, name='OutLayer')(decoder_outputs)
    model = tf.keras.models.Model([cond_inputs, inputs], decoder_outputs)

    model.summary()
    return model

def create_model_S4D(cond_dim, input_dim, units, mini_batch_size, b_size, stateful):
    T = input_dim
    D = cond_dim

    # Defining decoder inputs
    inputs = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, T), name='dec_input')

    decoder_outputs = tf.keras.layers.Dense(units // 2, name='LinearProjection')(
        inputs)

    # Defining encoder inputs

    cond_inputs = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, D), name='cond')

    decoder_outputs_ = tf.reshape(decoder_outputs, [b_size, mini_batch_size, units // 2])
    decoder_outputs = S4D(units//2, model_input_dims=units // 2, mini_batch_size=mini_batch_size, batch_size=b_size, hippo=True, stateful=stateful)(decoder_outputs_)

    decoder_outputs = tf.keras.layers.Dense(units // 2, activation='tanh', name='NonlinearDenseLayer')(
        decoder_outputs)

    decoder_outputs = FiLM(units // 2)(decoder_outputs, cond_inputs)

    decoder_outputs = tf.keras.layers.Dense(1, name='OutLayer')(decoder_outputs)
    model = tf.keras.models.Model([cond_inputs, inputs], decoder_outputs)

    model.summary()
    return model



def create_model_ED(cond_dim, input_dim, units, mini_batch_size, b_size, stateful):
    T = input_dim  # time window
    D = cond_dim
    cond_inputs = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, D), name='cond')

    inputs = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, T), name='inputs')

    decoder_outputs, decoder_inputs = ED_sharing_state(b_size=b_size, units=units, mini_batch_size=mini_batch_size, input_dim=T, stateful=stateful)(inputs)

    decoder_outputs = tf.keras.layers.Dense(units//2, name='Linear')(decoder_outputs)

    decoder_outputs = FiLM(units // 2)(decoder_outputs, cond_inputs)

    decoder_outputs = tf.keras.layers.Dense(1, name='OutLayer')(decoder_outputs)
    model = tf.keras.models.Model([cond_inputs, inputs], decoder_outputs)
    model.summary()

    return model


def create_model_LSTM(cond_dim, input_dim, units, mini_batch_size, b_size, stateful):
    T = input_dim
    D = cond_dim


    # Defining decoder inputs
    inputs = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, T), name='dec_input')
    decoder_outputs = tf.keras.layers.Dense(units//2, input_shape=(b_size, mini_batch_size, T), name='LinearProjection')(inputs)

    # Defining encoder inputs
    cond_inputs = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, D), name='cond')

    #decoder_outputs = tf.expand_dims(decoder_outputs, axis=1)

    decoder_outputs = tf.keras.layers.LSTM(units, stateful=stateful, return_sequences=True, return_state=False, name='LSTM')(
        decoder_outputs)
    decoder_outputs = tf.keras.layers.Dense(units//2, name='Linear')(decoder_outputs)

    decoder_outputs = FiLM(units // 2)(decoder_outputs, cond_inputs)

    decoder_outputs = tf.keras.layers.Dense(1, name='OutLayer')(decoder_outputs)
    model = tf.keras.models.Model([cond_inputs, inputs], decoder_outputs)

    model.summary()
    return model

def create_model_LRU(cond_dim, input_dim, units, mini_batch_size, b_size, stateful):
    T = input_dim
    D = cond_dim

    # Defining decoder inputs
    inputs = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, T), name='dec_input')
    decoder_outputs_ = tf.keras.layers.Dense(units // 2, batch_input_shape=(b_size, mini_batch_size, T), name='LinearProjection')(
        inputs)
    decoder_outputs = tf.reshape(decoder_outputs_, [b_size, mini_batch_size, units // 2])

    # Defining encoder inputs
    cond_inputs = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, D), name='cond')

    decoder_outputs = LRU(batch_size=b_size, model_states=units, input_dim=units // 2, stateful=stateful)(decoder_outputs)
    decoder_outputs = tf.keras.layers.Dense(units // 2, activation='tanh', name='NonlinearDenseLayer')(
        decoder_outputs)

    decoder_outputs = FiLM(units // 2)(decoder_outputs, cond_inputs)

    decoder_outputs = tf.keras.layers.Dense(1, name='OutLayer')(decoder_outputs)
    model = tf.keras.models.Model([cond_inputs, inputs], decoder_outputs)
    model.summary()
    return model