import tensorflow as tf
import math
import tensorflow_probability as tfp
import numpy as np
parallel_scan = tfp.math.scan_associative
from einops import repeat

class GLU(tf.keras.layers.Layer):
    """
    Gated Linear Unit
        :param in_size: input size
        :param bias: if use bias 
        :param dim: dimension for the split
    """
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

### from https://github.com/Gothos/LRU-pytorch/blob/main/LRU_pytorch/LRU.py
class LRU(tf.keras.layers.Layer):
    def __init__(self, N, H, r_min=0, r_max=1, max_phase=6.283):
        super(LRU, self).__init__()
        self.N = N
        self.H = H
        self.r_min = r_min
        self.r_max = r_max
        self.max_phase = max_phase
        #self.lru_parameters = self.init_lru_parameters()
        self.init_lru_parameters()

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

        self.B = tf.Variable(B, trainable=True)
        self.C = tf.Variable(C, trainable=True)
        self.D = tf.Variable(D, trainable=True)
        self.nu_log = tf.Variable(nu_log, trainable=True)
        self.theta_log = tf.Variable(theta_log, trainable=True)
        self.gamma_log = tf.Variable(gamma_log, trainable=True)

        self.state = tf.Variable(tf.complex(tf.zeros(self.N), tf.zeros(self.N)), trainable=False)

        #return nu_log, theta_log, B, C, D, gamma_log

    def reset_states(self):
        self.state = tf.Variable(tf.complex(tf.zeros(self.N), tf.zeros(self.N)), trainable=False)

    def binary_operator_diag(self, element_i, element_j):
        a_i, bu_i = element_i
        a_j, bu_j = element_j
        return a_j * a_i, a_j * bu_i + bu_j

    def call(self, input_sequence):

        # print(input_sequence.shape)
        #nu_log, theta_log, B, C, D, gamma_log = self.lru_parameters
        B = self.B
        C = self.C
        D = self.D
        nu_log = self.nu_log
        theta_log = self.theta_log
        gamma_log = self.gamma_log

        # Materializing the diagonal of Lambda and projections
        Lambda = tf.math.exp(tf.complex(-tf.math.exp(nu_log), tf.math.exp(theta_log)))
        exp_gamma_log = tf.math.exp(tf.complex(tf.zeros_like(gamma_log), gamma_log))
        B_norm = B * tf.expand_dims(exp_gamma_log, axis=-1)

        # Running the LRU + output projection
        Lambda_reshaped = tf.expand_dims(Lambda, axis=0)
        Lambda_reshaped = Lambda_reshaped * self.state ####
        Lambda_elements = tf.repeat(Lambda_reshaped, repeats=input_sequence.shape[0], axis=0)

        # Converting real input sequences to a complex form
        if input_sequence.dtype.is_complex:
            input_sequence = input_sequence
        else:
            input_sequence = tf.complex(input_sequence, tf.zeros_like(input_sequence))

        input_sequence_reshaped = tf.expand_dims(input_sequence, axis=-1)
        Bu_elements = tf.vectorized_map(lambda u: tf.linalg.matmul(B_norm, u), input_sequence_reshaped)
        Bu_elements = tf.squeeze(Bu_elements, axis=-1)

        #Lambda_elements = Lambda_elements * self.state

        elements = (Lambda_elements, Bu_elements)
        _, inner_states = parallel_scan(self.binary_operator_diag, elements)

        #self.state = (Lambda * self.state + B_norm @ step)
        self.state.assign(inner_states[-1])
        D = tf.cast(D, tf.complex64)
        y = tf.vectorized_map(lambda args: tf.math.real(tf.linalg.matvec(C, args[0])) + tf.math.real(D * args[1]),
                              (inner_states, input_sequence))
        return y



### from https://github.com/state-spaces/s4/blob/main/models/s4/
        
class S4DKernel(tf.keras.layers.Layer):
    """Generate convolution kernel from diagonal SSM parameters.
        A: (S, N) diagonal matrix
        B: (S, N)
        C: (C, H, N)
        dt: (H) timescale per feature

        Dimensions:
        N (or d_state): state size
        H (or d_model): total SSM copies
        S (or n_ssm): number of trainable copies of (A, B, dt); must divide H
        C (or channels): system is 1-dim to C-dim

        The forward pass of this Module returns a tensor of shape (C, H, L)

    """

    def __init__(self,  N=64, d_model=1, dt_min=0.001, dt_max=0.1, b_size=600, hippo=False):
        super(S4DKernel, self).__init__()
        self.N = N
        # Generate dt
        self.H = d_model
        #self.log_dt = tf.random.uniform([self.H]) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = tf.random.uniform((self.H,), minval=tf.math.log(dt_min), maxval=tf.math.log(dt_max))

        if hippo:
            A, A_imag, _, B = hippo_initializer(self.N)
            A = tf.cast(A, dtype=tf.float32)
            B = tf.reshape(B, [1, self.H, self.N])
            #log_A_real = tf.cast(tf.math.log(tf.reshape(A, [self.H, self.N])), dtype=tf.float32)
            A_imag = tf.cast(tf.reshape(A_imag, [self.H, self.N]), dtype=tf.float32)
            self.log_A_real = tf.Variable(A, name='log_A_real', trainable=False)
            self.A_imag = tf.Variable(A_imag, name='A_imag', trainable=False)

            B = tf.Variable(B, name='B', trainable=True)
            B = tf.tile(B, [1, 1, 1])
            self.B = tf.reshape(B, [-1, 1, B.shape[-1]])

        else:
            log_A_real = tf.math.log(tf.constant(0.5 * tf.ones((self.H, self.N))))
            A_imag = tf.constant(np.pi) * repeat(np.arange(N), 'n -> h n', h=self.H)
            self.log_A_real = tf.Variable(log_A_real, name='log_A_real', trainable=True)
            self.A_imag = tf.Variable(A_imag, name='A_imag', trainable=True)

            B = tf.Variable(0.5 * tf.ones((1, self.H, self.N)), name='B', trainable=True, dtype='float32')
            B = tf.tile(B, [1, 1, 1])
            self.B = tf.reshape(B, [-1, 1, B.shape[-1]])

        C = tf.complex(tf.random.normal([self.H, self.N]), tf.random.normal([self.H, self.N]))
        self.C = tf.Variable(tf.cast(C, dtype=tf.float32), trainable=True)
        self.log_dt = tf.Variable(self.log_dt, trainable=True)

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
        B = tf.cast(self.B, dtype=tf.complex64)
        #dt = tf.expand_dims(dt, axis=1)
        s = self.state / tf.cast(dt, dtype=tf.complex64)
        s = s * dtA * tf.exp(dtA) / (tf.exp(dtA) - 1.)
        B = tf.concat([s, B], axis=0)  # (1+B H N)
        # # Combine B and C
        C = B[:, None, :, :] * C
        C = tf.reshape(C, [-1, self.H, self.N])



        C = C * (tf.exp(dtA) - 1.) / A
        ##K = log_vandermonde(C, dtA, L)  # (H L)
        K = tf.exp(tf.expand_dims(dtA, axis=-1) * tf.cast(tf.range(L), dtype=tf.complex64))# (H N L)
        K = tf.tensordot(C, K, axes=[[-1], [-2]])
        #K = tf.math.real(2 * tf.einsum('hn, hnl -> hl', C, K))
        #K = 2 * tf.reduce_sum(C * tf.exp(K), axis=1, keepdims=False)


        ####States
        K = tf.reshape(K, [-1, 1, self.H, L])  # (1+B C H L)

        state = tf.cast(K[:-1, 0, :, :], dtype=tf.complex64)
        self.state.assign(state)
        K = K[-1, 0, :, :]  # (C H L)

        return tf.math.real(K)

    def reset_states(self):
        self.state = tf.Variable(tf.zeros((self.b_size, self.H, self.N), dtype=tf.complex64), name='state', trainable=False)

class S4D(tf.keras.layers.Layer):
    def __init__(self, d_state=64, d_model=1, transposed=True, hippo=False, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = tf.Variable(tf.random.normal([self.h]), dtype='float32')

        # SSM Kernel
        self.kernel = S4DKernel(N=self.n, d_model=self.h, hippo=hippo, **kernel_args)

        # Pointwise
        self.activation = tf.keras.activations.gelu

        #self.conv = tf.keras.layers.Conv1D(2 * self.h, kernel_size=1, dtype='float32')
        self.glu = GLU(in_size=2*self.n, dim=-1)

    def reset_states(self):
        self.kernel.reset_states()

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

        #y = self.activation(y)
        #y = tf.expand_dims(y, axis=-1)
        #y = self.conv(y)
        y = self.glu(y)
        return y


def hippo_initializer(N):
    Lambda, P, B, _ = make_DPLR_HiPPO(N)
    return tf.math.real(Lambda), tf.math.imag(Lambda), P, B

def make_DPLR_HiPPO(N):
    """Diagonalize NPLR representation"""
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    # Check skew symmetry
    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)
    # assert np.allclose(Lambda_real, S_diag, atol=1e-3)

    # Diagonalize S to V \Lambda V^*
    #Lambda_imag, V = tf.linalg.eigh(S * -1j)
    S = tf.complex(tf.cast(0., dtype=tf.float64), -S)
    Lambda_imag, V = tf.linalg.eigh(S)

    Lambda_imag = tf.math.imag(Lambda_imag)
    #P = V.conj().T @ P
    #B = V.conj().T @ B
    P = tf.linalg.matrix_transpose(V) @ tf.cast(tf.linalg.adjoint(P[np.newaxis, :]), dtype=tf.complex128)
    B = tf.linalg.matrix_transpose(V) @ tf.cast(tf.linalg.adjoint(B[np.newaxis, :]), dtype=tf.complex128)
    Lambda = tf.complex(Lambda_real, Lambda_imag)
    return Lambda, P, B, V

def make_NPLR_HiPPO(N):
    # Make -HiPPO
    nhippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return nhippo, P, B

def make_HiPPO(N):
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A
