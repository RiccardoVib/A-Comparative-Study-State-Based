import tensorflow as tf
import math
import numpy as np
from einops import repeat

#### from https://github.com/state-spaces/mamba & https://github.com/PeaBrane/mamba-tiny



class S6(tf.keras.layers.Layer):
    def __init__(self, model_input_dims, model_states, **kwargs):
        super(S6, self).__init__(**kwargs)
        self.model_input_dims = model_input_dims
        self.model_states = model_states
        self.delta_t_rank = math.ceil(model_input_dims / 2)  # 16

        self.state = tf.Variable(tf.zeros((1, self.model_internal_dim, self.model_states), dtype=tf.float32), name='state', trainable=False)

        self.x_projection = tf.keras.layers.Dense(self.delta_t_rank + self.model_states * 2, use_bias=False)

        self.delta_t_projection = tf.keras.layers.Dense(self.model_input_dims,
                                               input_shape=(self.delta_t_rank,), use_bias=True)

        self.A = repeat(
            tf.range(1, self.model_states + 1, dtype=tf.float32),
            'n -> d n', d=self.model_input_dims)

        self.A_log = tf.Variable(
            tf.math.log(self.A),
            trainable=True, dtype=tf.float32,
            name=f"SSM_A_log_0")

        self.D = tf.Variable(
            np.ones(self.model_input_dims),
            trainable=True, dtype=tf.float32,
            name=f"SSM_D_0")

        self.out_projection = tf.keras.layers.Dense(
            self.model_input_dims,
            input_shape=(self.model_input_dims,),
            use_bias=True)

    def call(self, x):
        y = self.ssm(x)
        return self.out_projection(y)

    def ssm(self, x):
       
        (d_in, n) = self.A_log.shape


        A = -tf.exp(tf.cast(self.A_log, tf.float32))  # shape -> (d_in, n)
        D = tf.cast(self.D, tf.float32)

        x_dbl = self.x_projection(x)  # shape -> (batch, seq_len, delta_t_rank + 2*n)

        (delta, B, C) = tf.split(
            x_dbl,
            num_or_size_splits=[self.delta_t_rank, n, n],
            axis=-1)  # delta.shape -> (batch, seq_len) & B, C shape -> (batch, seq_len, n)

        delta = tf.nn.softplus(self.delta_t_projection(delta))  # shape -> (batch, seq_len, model_input_dim)

        y, last_state = selective_scan(x, delta, A, B, C, D, self.state)
        self.state.assign(last_state)
        return y

def selective_scan(u, delta, A, B, C, D, last_state):
    # first step of A_bar = exp(ΔA), i.e., ΔA
    dA = tf.einsum('bld,dn->bldn', delta, A)
    dB_u = tf.einsum('bld,bld,bln->bldn', delta, u, B)

    dA_cumsum = tf.concat([last_state[:, np.newaxis, :, :], dA[:, 1:]], axis=1)  ##### add state in the first spot since starting with state=0

    dA_cumsum = tf.exp(dA_cumsum)

    xs = []

    for i in range(1):
        last_state = dA_cumsum[:, i] * last_state + dB_u[:, i]
        xs.append(last_state)
    # Cumulative sum along all the input tokens, parallel prefix sum,
    # calculates dA for all the input tokens parallely
    #dA_cumsum = tf.math.cumsum(dA_cumsum, axis=1)

    # second step of A_bar = exp(ΔA), i.e., exp(ΔA)
    #dA_cumsum = tf.exp(dA_cumsum)
    #dA_cumsum = tf.reverse(dA_cumsum, axis=[1])  # Flip back along axis 1

    #x = dB_u * dA_cumsum
    # 1e-12 to avoid division by 0
    #x = tf.math.cumsum(x, axis=1) / (dA_cumsum + 1e-12)

    #if stateful == True:
    #    last_state = x[:, -1:]

    x = tf.stack(xs, axis=1)
    y = tf.einsum('bldn,bln->bld', x, C)

    return y + u * D, last_state