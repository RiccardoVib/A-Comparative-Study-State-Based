# Copyright (C) 2025 Riccardo Simionato, University of Oslo
# Inquiries: riccardo.simionato.vib@gmail.com.com
#
# This code is free software: you can redistribute it and/or modify it under the terms
# of the GNU Lesser General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Less General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this code.
# If not, see <http://www.gnu.org/licenses/>.
#
# If you use this code or any part of it in any program or publication, please acknowledge
# its authors by adding a reference to this publication:
#
# R. Simionato, 2025, "A Comparative Study of State-based Neural Networks for Virtual Analog Audio Effects Modeling" in EURASIP Journal on Audio, Speech, and Music Processing, 2025.
import tensorflow as tf

class ED_sharing_state(tf.keras.layers.Layer):
    def __init__(self, b_size, units, input_dim, mini_batch_size, stateful, type=tf.float32):
        super(ED_sharing_state, self).__init__()
        self.b_size = b_size
        self.mini_batch_size = mini_batch_size
        self.units = units
        self.type = type
        self.input_dim = input_dim
        self.stateful = stateful

        self.h_ = tf.Variable(tf.zeros((b_size, units // 2)), trainable=False)
        self.c_ = tf.Variable(tf.zeros((b_size, units // 2)), trainable=False)

        self.conv_h = tf.keras.layers.Conv2D(units // 2, kernel_size=[1, input_dim-1], name='Conv_h')
        self.conv_c = tf.keras.layers.Conv2D(units // 2, kernel_size=[1, input_dim-1], name='Conv_c')
        self.dense = tf.keras.layers.Dense(units//2, input_shape=(b_size, 1), name='LinearProjection')
        self.decoder = tf.keras.layers.LSTM(units//2, return_sequences=True, return_state=True, stateful=stateful, name='LSTM_decoder')
        #x = tf.Variable(tf.zeros((b_size, mini_batch_size, 64)), trainable=False)
    def call(self, x):
        encoder_inputs, decoder_inputs = tf.split(x, [self.input_dim - 1, 1], axis=2)
        h = self.conv_h(tf.expand_dims(encoder_inputs[:,0:1,:], axis=-1))
        c = self.conv_c(tf.expand_dims(encoder_inputs[:,0:1,:], axis=-1))
        h = tf.squeeze(h, axis=2)
        c = tf.squeeze(c, axis=2)

        decoder_outputs = self.dense(decoder_inputs)

        h1, c1 = self.h_, self.c_

        h1 = tf.multiply(tf.sigmoid(h1), h[:, 0, :])
        c1 = tf.multiply(tf.sigmoid(c1), c[:, 0, :])

        outs, h1, c1 = self.decoder(decoder_outputs, initial_state=[h1, c1])

        self.h_.assign(h1)
        self.c_.assign(c1)

        return outs, decoder_inputs


class FiLM(tf.keras.layers.Layer):
    def __init__(self, in_size, bias=True, dim=-1, **kwargs):
        super(FiLM, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.in_size = in_size
        self.dense = tf.keras.layers.Dense(self.in_size * 2, use_bias=bias)
        self.glu = GLU(in_size=self.in_size)

    def call(self, x, c):
        c = self.dense(c)
        a, b = tf.split(c, 2, axis=self.dim)
        x = tf.multiply(a, x)
        x = tf.add(x, b)
        x = self.glu(x)
        return x

class GLU(tf.keras.layers.Layer):
    def __init__(self, in_size, bias=True, dim=-1, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.in_size = in_size
        self.dense = tf.keras.layers.Dense(self.in_size * 2, use_bias=bias, dtype='float32')

    def call(self, x):
        x = self.dense(x)
        out, gate = tf.split(x, 2, axis=self.dim)
        gate = tf.keras.activations.softsign(gate)
        x = tf.multiply(out, gate)
        return x