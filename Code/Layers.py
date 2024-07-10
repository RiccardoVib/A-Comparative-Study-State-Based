import tensorflow as tf
import math
import tensorflow_probability as tfp
import numpy as np
parallel_scan = tfp.math.scan_associative
from einops import repeat

class GLU(tf.keras.layers.Layer):
    def __init__(self, in_size, bias=True, dim=-1, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.in_size = in_size
        self.dense = tf.keras.layers.Dense(self.in_size * 2, use_bias=bias)

    def call(self, x):
        x = self.dense(x)
        out, gate = tf.split(x, 2, axis=self.dim)
        gate = tf.keras.activations.softsign(gate)
        x = tf.multiply(out, gate)
        return x


class LRU(tf.keras.layers.Layer):
    def __init__(self, N, H, r_min=0, r_max=1, max_phase=6.283):
        super(LRU, self).__init__()
        self.N = N
        self.H = H
        self.r_min = r_min
        self.r_max = r_max
        self.max_phase = max_phase
        self.lru_parameters = self.init_lru_parameters()

    def init_lru_parameters(self):
        # N: state dimension, H: model dimension
        # Initializating Lambda
        u1 = tf.random.uniform(shape=(self.N,))
        u2 = tf.random.uniform(shape=(self.N,))
        nu_log = tf.math.log(-0.5 * tf.math.log(u1 * (self.r_max ** 2 - self.r_min ** 2) + self.r_min ** 2))
        theta_log = tf.math.log(u2 * self.max_phase)

        # Glorot initialized Input/Output projection matrices
        B = tf.complex(tf.random.normal(shape=(self.N, self.H)) / math.sqrt(2 * self.H),
                       tf.random.normal(shape=(self.N, self.H)) / math.sqrt(2 * self.H))
        C = tf.complex(tf.random.normal(shape=(self.H, self.N)) / math.sqrt(self.N),
                       tf.random.normal(shape=(self.H, self.N)) / math.sqrt(self.N))
        D = tf.random.normal(shape=(self.H,))

        # Normalization factor
        diag_lambda = tf.math.exp(tf.complex(-tf.math.exp(nu_log), tf.math.exp(theta_log)))
        gamma_log = tf.math.log(tf.math.sqrt(1 - tf.math.abs(diag_lambda) ** 2))

        return nu_log, theta_log, B, C, D, gamma_log

    def binary_operator_diag(self, element_i, element_j):
        a_i, bu_i = element_i
        a_j, bu_j = element_j
        return a_j * a_i, a_j * bu_i + bu_j

    def call(self, input_sequence):

        # print(input_sequence.shape)
        nu_log, theta_log, B, C, D, gamma_log = self.lru_parameters
        # Materializing the diagonal of Lambda and projections
        Lambda = tf.math.exp(tf.complex(-tf.math.exp(nu_log), tf.math.exp(theta_log)))
        exp_gamma_log = tf.math.exp(tf.complex(tf.zeros_like(gamma_log), gamma_log))
        B_norm = B * tf.expand_dims(exp_gamma_log, axis=-1)

        # Running the LRU + output projection
        Lambda_reshaped = tf.expand_dims(Lambda, axis=0)
        Lambda_elements = tf.repeat(Lambda_reshaped, repeats=input_sequence.shape[0], axis=0)

        # Converting real input sequences to a complex form
        if input_sequence.dtype.is_complex:
            input_sequence = input_sequence
        else:
            input_sequence = tf.complex(input_sequence, tf.zeros_like(input_sequence))

        input_sequence_reshaped = tf.expand_dims(input_sequence, axis=-1)
        Bu_elements = tf.vectorized_map(lambda u: tf.linalg.matmul(B_norm, u), input_sequence_reshaped)
        Bu_elements = tf.squeeze(Bu_elements, axis=-1)

        elements = (Lambda_elements, Bu_elements)
        _, inner_states = parallel_scan(self.binary_operator_diag, elements)
        D = tf.cast(D, tf.complex64)
        y = tf.vectorized_map(lambda args: tf.math.real(tf.linalg.matvec(C, args[0])) + tf.math.real(D * args[1]),
                              (inner_states, input_sequence))
        return y


### from https://github.com/state-spaces/s4/blob/main/models/s4/
        
class S4DKernel(tf.keras.layers.Layer):
    def __init__(self,  N=64, d_model=1, dt_min=0.001, dt_max=0.1, b_size=600):
        super(S4DKernel, self).__init__()
        self.N = N
        # Generate dt
        self.H = d_model
        self.log_dt = tf.random.uniform((self.H,), minval=tf.math.log(dt_min), maxval=tf.math.log(dt_max))

        #self.B = tf.Variable(0.5 * tf.ones((self.H, self.N//2)), trainable=True, dtype='float32')

        C = tf.complex(tf.random.normal([self.H, self.N // 2]), tf.random.normal([self.H, self.N // 2]))
        self.C = tf.Variable(tf.cast(C, dtype=tf.float32), trainable=True)

        self.log_dt = tf.Variable(self.log_dt, trainable=True)

        log_A_real = tf.math.log(tf.constant(0.5 * tf.ones((self.H, self.N//2))))
        A_imag = tf.constant(np.pi) * repeat(np.arange(N//2), 'n -> h n', h=self.H)

        self.log_A_real = tf.Variable(log_A_real, trainable=True)
        self.A_imag = tf.Variable(A_imag, trainable=True)

        self.b_size = b_size
        self.reset_states()

    def call(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = tf.exp(self.log_dt)  # (H)
        C = tf.cast(self.C, dtype=tf.complex64)  # (H N)

        A = tf.dtypes.complex(-tf.exp(self.log_A_real), self.A_imag)

        # Vandermonde multiplication
        dt = tf.expand_dims(dt, axis=-1)
        dtA = tf.cast(tf.math.real(A) * dt, dtype=tf.complex64)   # (H N)

        ####States
        # Augment B with state
        # s = tf.math.real(self.state) / dt
        # s = s * dtA * tf.exp(dtA) / (tf.exp(dtA) - 1.)
        # B = tf.concat([s, self.B], axis=-3)  # (1+B H N)
        # # Combine B and C
        # C = tf.reshape(B[:, None, :, :] * C, [-1, self.H, self.N])

        K = tf.expand_dims(dtA, axis=-1) * tf.cast(tf.range(L), dtype=tf.complex64)  # (H N L)
        C = C * (tf.exp(dtA) - 1.) / A
        K = tf.math.real(2 * tf.einsum('hn, hnl -> hl', C, tf.exp(K)))
        #K = 2 * tf.reduce_sum(C * tf.exp(K), axis=1, keepdims=False)

        ####States
        # K = tf.reshape(K, [-1, 1, self.H, L])  # (1+B C H L)
        # self.state = K[:-1, :, :, :]  # (B C H L)
        # K = K[-1, :, :, :]  # (C H L)

        return K

    def reset_states(self):
        self.state = tf.zeros((self.H, self.N//2), dtype=tf.complex64)

class S4D(tf.keras.layers.Layer):
    def __init__(self, d_state=64, d_model=1, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = tf.Variable(tf.random.normal([self.h]), dtype='float32')

        # SSM Kernel
        self.kernel = S4DKernel(N=self.n, d_model=self.h, **kernel_args)

        # Pointwise
        self.activation = tf.keras.activations.gelu
        self.glu = GLU(in_size=2*self.n, dim=-1)

    def call(self, u):  # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        L = u.shape[-1]

        # Compute SSM Kernel
        k = self.kernel(L=L)  # (H L)
        # Convolution
        k_f = tf.signal.rfft(k, fft_length=[2 * L])  # (H L)
        u_f = tf.signal.rfft(u, fft_length=[2 * L])  # (B H L)
        y = tf.signal.irfft((u_f * k_f), fft_length=[2 * L])[..., :L]  # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * tf.expand_dims(self.D, axis=-1)
        y = self.glu(y)

        return y
