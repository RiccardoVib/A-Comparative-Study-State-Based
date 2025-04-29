import tensorflow as tf
import math
import numpy as np
from einops import repeat


class S6(tf.keras.layers.Layer):
    def __init__(self, model_input_dims, model_states, batch_size, stateful, **kwargs):
        super(S6, self).__init__(**kwargs)
        self.model_input_dims = model_input_dims
        self.model_states = model_states
        self.stateful = stateful
        self.batch_size = batch_size

        self.delta_t_rank = math.ceil(model_input_dims / 2)  # 16

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
            input_shape=(self.model_input_dims,), trainable=False)

        self.reset_states()

    def reset_states(self):
        self.state = tf.Variable(
            tf.zeros((self.batch_size, self.model_input_dims, self.model_states), dtype=tf.float32), name='state',
            trainable=False)

    def call(self, x):

        last_state = self.state[:self.batch_size]
        res_state = self.state[self.batch_size:]

        y, y_state = self.ssm(x, last_state=last_state, stateful=self.stateful)

        if self.stateful:
            self.state.assign(tf.concat([y_state, res_state], axis=0))

        return self.out_projection(y)

    def ssm(self, x, last_state, stateful):

        (d_in, n) = self.A_log.shape

        A = -tf.exp(tf.cast(self.A_log, tf.float32))  # shape -> (d_in, n)
        D = tf.cast(self.D, tf.float32)

        x_dbl = self.x_projection(x)  # shape -> (batch, seq_len, delta_t_rank + 2*n)

        (delta, B, C) = tf.split(
            x_dbl,
            num_or_size_splits=[self.delta_t_rank, n, n],
            axis=-1)  # delta.shape -> (batch, seq_len) & B, C shape -> (batch, seq_len, n)

        delta = tf.nn.softplus(self.delta_t_projection(delta))  # shape -> (batch, seq_len, model_input_dim)

        return selective_scan(x, delta, A, B, C, D, last_state, stateful)


def selective_scan(u, delta, A, B, C, D, last_state, stateful):
    dA = tf.einsum('bld,dn->bldn', delta, A)
    dB_u = tf.einsum('bld,bld,bln->bldn', delta, u, B)

    dA_cumsum = tf.pad(dA[:, 1:], [[0, 0], [1, 0], [0, 0], [0, 0]])

    dA_cumsum = tf.math.cumsum(dA_cumsum, axis=1)
    dA_cumsum = tf.exp(dA_cumsum)

    x = dB_u / (dA_cumsum + 1e-12)
    x = tf.math.cumsum(x, axis=1) * dA_cumsum

    if stateful:
        dA_cumsum_l = tf.math.cumsum(dA, axis=1)
        dA_cumsum_l = tf.exp(dA_cumsum_l)
        dA_cumsum_l *= tf.expand_dims(last_state, axis=1)
        x = x + dA_cumsum_l

    last_state = x[:, -1]
    y = tf.einsum('bldn,bln->bld', x, C)

    return y + u * D, last_state