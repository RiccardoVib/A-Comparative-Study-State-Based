import tensorflow as tf
import math
from einops import repeat


def selective_scan(Lambda_elements, Bu_elements, last_state, stateful):

    dA_cumsum = tf.pad(
        Lambda_elements[:, 1:], [[0, 0], [1, 0], [0, 0], [0, 0]])

    # Cumulative sum along all the input tokens, parallel prefix sum,
    # calculates dA for all the input tokens parallely
    dA_cumsum = tf.math.cumsum(dA_cumsum, axis=1)
    dA_cumsum = tf.clip_by_value(dA_cumsum, clip_value_min=-0.9, clip_value_max=0.9)
    # second step of A_bar = exp(ΔA), i.e., exp(ΔA)
    dA_cumsum = tf.exp(dA_cumsum)

    x = Bu_elements * dA_cumsum
    # 1e-12 to avoid division by 0
    x = tf.math.cumsum(x, axis=1) / (dA_cumsum + 1e-12)

    if stateful:
        dA_cumsum_l = tf.math.cumsum(Lambda_elements, axis=1)
        dA_cumsum_l = tf.clip_by_value(dA_cumsum_l, clip_value_min=-0.5, clip_value_max=0.5)
        dA_cumsum_l = tf.exp(dA_cumsum_l)
        dA_cumsum_l *= tf.expand_dims(last_state, axis=1)
        x = x + dA_cumsum_l

    return x

class LRU(tf.keras.layers.Layer):
    def __init__(self, batch_size, model_states, input_dim, r_min=0.9, r_max=0.999, max_phase=6.283, stateful=False):#0.9, 0.999
        super(LRU, self).__init__()
        self.model_states = model_states
        self.input_dim = input_dim
        self.r_min = r_min
        self.r_max = r_max
        self.max_phase = max_phase
        self.batch_size = batch_size
        self.stateful = stateful
        self.init_lru_parameters()

    def init_lru_parameters(self):
        # N: state dimension, H: model dimension
        # Initializing Lambda
        u1 = tf.random.uniform(shape=(self.model_states,))
        u2 = tf.random.uniform(shape=(self.model_states,))
        nu_log = tf.math.log(-0.5 * tf.math.log(u1 * (self.r_max ** 2 - self.r_min ** 2) + self.r_min ** 2))
        theta_log = tf.math.log(u2 * self.max_phase)

        # Glorot initialized Input/Output projection matrices
        B = tf.complex(tf.random.normal(shape=(self.model_states, self.input_dim)) / math.sqrt(2 * self.input_dim),
                       tf.random.normal(shape=(self.model_states, self.input_dim)) / math.sqrt(2 * self.input_dim))
        C = tf.random.normal(shape=(self.input_dim, self.model_states)) / math.sqrt(self.model_states)
        D = tf.random.normal(shape=(self.input_dim,))

        # Normalization factor
        diag_lambda = tf.math.exp(tf.complex(-tf.math.exp(nu_log), tf.math.exp(theta_log)))
        gamma_log = tf.math.log(tf.math.sqrt(1 - tf.math.abs(diag_lambda) ** 2))

        self.B_real = tf.Variable(tf.math.real(B), trainable=True)
        self.B_imag = tf.Variable(tf.math.imag(B), trainable=True)

        self.C_real = tf.Variable(C, trainable=True)
        self.D = tf.Variable(D, trainable=True)
        self.nu_log = tf.Variable(nu_log, trainable=True)
        self.theta_log = tf.Variable(theta_log, trainable=True)
        self.gamma_log = tf.Variable(gamma_log, trainable=True)

        self.reset_states()
        #input_sequence = tf.Variable(tf.random.normal([self.batch_size, 2048, self.input_dim]), dtype='float32')

    def reset_states(self):
        self.state_real = tf.Variable(tf.zeros((self.batch_size+1, self.input_dim, self.model_states), dtype=tf.float32), trainable=False)

    def call(self, input_sequence):

        B = tf.complex(self.B_real, self.B_imag)
        C = self.C_real

        last_state_r = self.state_real[:self.batch_size]
        res_state_r = self.state_real[self.batch_size:]

        D = self.D
        nu_log = self.nu_log
        theta_log = self.theta_log
        gamma_log = self.gamma_log

        # Materializing the diagonal of Lambda and projections
        Lambda = tf.math.exp(tf.complex(-tf.math.exp(nu_log), tf.math.exp(theta_log)))
        exp_gamma_log = tf.math.exp(tf.complex(tf.zeros_like(gamma_log), gamma_log))
        B_norm = B * tf.expand_dims(exp_gamma_log, axis=-1)

        # Running the LRU + output projection
        Lambda_reshaped = repeat(Lambda, 'n -> d n', d=self.input_dim)
        Lambda_elements = repeat(Lambda_reshaped, 'd n -> b d n', b=self.batch_size)
        Lambda_elements = repeat(Lambda_elements, 'b d n -> b l d n', l=input_sequence.shape[1])

        B_norm = tf.math.real(B_norm)
        Lambda_elements = tf.math.real(Lambda_elements)
        Bu_elements = tf.einsum('bld,dn->bldn', input_sequence, tf.reshape(B_norm, [self.input_dim, self.model_states]))

        inner_states = selective_scan(Lambda_elements, Bu_elements, last_state_r, self.stateful)
        last_state = inner_states[:, -1]

        y = tf.einsum('bldn,dn->bld', inner_states, C)
        y = y + input_sequence * D

        if self.stateful:
           self.state_real.assign(tf.concat([tf.math.real(last_state), res_state_r], axis=0))

        return y