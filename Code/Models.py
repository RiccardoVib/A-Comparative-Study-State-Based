import tensorflow as tf
from Layers import FiLM
from S4D import S4D
from LRU import LRU
from S6 import S6

def create_model_S6(D, T, units, b_size=2399):
    """ 
    S6 model
    :param T: input size
    :param D: number of conditioning parameters
    :param units: number of units
    """
    # Defining decoder inputs
    inputs = tf.keras.layers.Input(batch_shape=(b_size, T), name='dec_input')
    decoder_outputs = tf.keras.layers.Dense(units // 2, input_shape=(b_size, T), name='LinearProjection')(
        inputs)
    decoder_outputs = tf.reshape(decoder_outputs, [b_size, 1, units // 2])

    # Defining encoder inputs
    cond_inputs = tf.keras.layers.Input(batch_shape=(b_size, D), name='cond')

    decoder_outputs = S6(model_input_dims=units//2, model_states=units//2)(decoder_outputs)[:, 0, :]

    decoder_outputs = tf.keras.layers.Dense(units // 2, activation='tanh', name='NonlinearDenseLayer')(
        decoder_outputs)

    decoder_outputs = FiLM(units // 2)(decoder_outputs, cond_inputs)

    decoder_outputs = tf.keras.layers.Dense(1, name='OutLayer')(decoder_outputs)
    model = tf.keras.models.Model([cond_inputs, inputs], decoder_outputs)

    model.summary()
    return model
    
def create_model_S4D(D, T, units, b_size=2399):
    """ 
    S4D model
    :param T: input size
    :param D: number of conditioning parameters
    :param units: number of units
    """
    # Defining decoder inputs
    inputs = tf.keras.layers.Input(batch_shape=(b_size, T), name='dec_input')
    decoder_outputs_ = tf.keras.layers.Dense(units // 2, input_shape=(b_size, T), name='LinearProjection')(
        inputs)
    decoder_outputs = tf.reshape(decoder_outputs_, [b_size, 1, units // 2])

    # Defining encoder inputs

    cond_inputs = tf.keras.layers.Input(batch_shape=(b_size, D), name='cond')

    decoder_outputs = S4D(units//2, b_size=b_size)(decoder_outputs)[:, 0, :]

    decoder_outputs = tf.keras.layers.Dense(units // 2, activation='tanh', name='NonlinearDenseLayer')(
        decoder_outputs)

    decoder_outputs = FiLM(units // 2)(decoder_outputs, cond_inputs)

    decoder_outputs = tf.keras.layers.Dense(1, name='OutLayer')(decoder_outputs)
    model = tf.keras.models.Model([cond_inputs, inputs], decoder_outputs)

    model.summary()
    return model



def create_model_ED(D, T, units, b_size=2399):
    """ 
    ED model
    :param T: input size
    :param D: number of conditioning parameters
    :param units: number of units
    """
    cond_inputs = tf.keras.layers.Input(batch_shape=(b_size, D), name='cond')

    inputs = tf.keras.layers.Input(batch_shape=(b_size, T, 1), name='inputs')

    decoder_outputs, decoder_inputs = ED_sharing_state(b_size, units, T)(inputs)

    decoder_outputs = tf.keras.layers.Dense(units//2, name='Linear')(decoder_outputs)

    decoder_outputs = FiLM(units // 2)(decoder_outputs, cond_inputs)

    decoder_outputs = tf.keras.layers.Dense(1, name='OutLayer')(decoder_outputs)
    model = tf.keras.models.Model([cond_inputs, inputs], decoder_outputs)
    model.summary()

    return model


def create_model_LSTM(D, T, units, b_size=2399):
    """ 
    LSTM model
    :param T: input size
    :param D: number of conditioning parameters
    :param units: number of units
    """


    # Defining decoder inputs
    inputs = tf.keras.layers.Input(batch_shape=(b_size, T), name='dec_input')
    decoder_outputs = tf.keras.layers.Dense(units//2, input_shape=(b_size, T), name='LinearProjection')(inputs)

    # Defining encoder inputs
    cond_inputs = tf.keras.layers.Input(batch_shape=(b_size, D), name='cond')

    decoder_outputs = tf.expand_dims(decoder_outputs, axis=1)

    decoder_outputs = tf.keras.layers.LSTM(units, stateful=True, return_sequences=False, return_state=False, name='LSTM')(
        decoder_outputs)
    decoder_outputs = tf.keras.layers.Dense(units//2, name='Linear')(decoder_outputs)

    decoder_outputs = FiLM(units // 2)(decoder_outputs, cond_inputs)

    decoder_outputs = tf.keras.layers.Dense(1, name='OutLayer')(decoder_outputs)
    model = tf.keras.models.Model([cond_inputs, inputs], decoder_outputs)

    model.summary()
    return model

def create_model_LRU(D, T, units, b_size=2399):
    """ 
    LRU model
    :param T: input size
    :param D: number of conditioning parameters
    :param units: number of units
    """
    # Defining decoder inputs
    inputs = tf.keras.layers.Input(batch_shape=(b_size, T), name='dec_input')
    decoder_outputs_ = tf.keras.layers.Dense(units // 2, batch_input_shape=(b_size, T), name='LinearProjection')(
        inputs)
    decoder_outputs = tf.reshape(decoder_outputs_, [b_size, units // 2])

    # Defining encoder inputs
    cond_inputs = tf.keras.layers.Input(batch_shape=(b_size, D), name='cond')

    decoder_outputs = LRU(N=units, H=units // 2)(decoder_outputs)  # units//4
    decoder_outputs = tf.keras.layers.Dense(units // 2, activation='tanh', name='NonlinearDenseLayer')(
        decoder_outputs)

    decoder_outputs = FiLM(units // 2)(decoder_outputs, cond_inputs)

    decoder_outputs = tf.keras.layers.Dense(1, name='OutLayer')(decoder_outputs)
    model = tf.keras.models.Model([cond_inputs, inputs], decoder_outputs)
    model.summary()
    return model
