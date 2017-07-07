import numpy as np
import tensorflow as tf

from tensorflow.python.util import nest
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell


def orthogonal(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)


def bn_lstm_identity_initializer(scale):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        '''Ugly cause LSTM params calculated in one matrix multiply'''
        size = shape[0]
        # gate (j) is identity
        t = np.zeros(shape)
        t[:, size:size * 2] = np.identity(size) * scale
        t[:, :size] = orthogonal([size, size])
        t[:, size * 2:size * 3] = orthogonal([size, size])
        t[:, size * 3:] = orthogonal([size, size])
        return tf.constant(t, dtype)

    return _initializer


def orthogonal_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        return tf.constant(orthogonal(shape), dtype)

    return _initializer


class IdentityLSTMCell(RNNCell):
    '''Vanilla LSTM implemented with same initializations as BN-LSTM'''

    def __init__(self, num_units):
        self.num_units = num_units

    @property
    def state_size(self):
        return (self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            # Keep W_xh and W_hh separate here as well to reuse initialization methods
            x_size = x.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh',
                                   [x_size, 4 * self.num_units],
                                   initializer=orthogonal_initializer())
            W_hh = tf.get_variable('W_hh',
                                   [self.num_units, 4 * self.num_units],
                                   initializer=bn_lstm_identity_initializer(0.95))
            bias = tf.get_variable('bias', [4 * self.num_units])

            # hidden = tf.matmul(x, W_xh) + tf.matmul(h, W_hh) + bias
            # improve speed by concat.
            concat = tf.concat(axis=1, values=[x, h])
            W_both = tf.concat(axis=0, values=[W_xh, W_hh])
            hidden = tf.matmul(concat, W_both) + bias

            i, j, f, o = tf.split(axis=1, num_or_size_splits=4, value=hidden)

            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)

            return new_h, (new_c, new_h)


class ZoneoutWrapper(RNNCell):
    """Operator adding zoneout to all states (states+cells) of the given cell."""

    def __init__(self, cell, zoneout_prob, is_training=True, seed=None):
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not an RNNCell.")
        if (isinstance(zoneout_prob, float) and
                not (zoneout_prob >= 0.0 and zoneout_prob <= 1.0)):
            raise ValueError("Parameter zoneout_prob must be between 0 and 1: %d"
                             % zoneout_prob)
        self._cell = cell
        self._zoneout_prob = zoneout_prob
        self._seed = seed
        self.is_training = tf.convert_to_tensor(is_training)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        if isinstance(self.state_size, tuple) != isinstance(self._zoneout_prob, tuple):
            raise TypeError("Subdivided states need subdivided zoneouts.")
        if isinstance(self.state_size, tuple) and len(tuple(self.state_size)) != len(tuple(self._zoneout_prob)):
            raise ValueError("State and zoneout need equally many parts.")
        output, new_state = self._cell(inputs, state, scope)
        if isinstance(self.state_size, tuple):

            def train():
                return tuple((1 - state_part_zoneout_prob) * tf.nn.dropout(
                    new_state_part - state_part, (1 - state_part_zoneout_prob), seed=self._seed) + state_part
                                  for new_state_part, state_part, state_part_zoneout_prob in
                                  zip(new_state, state, self._zoneout_prob))

            def test():
                return tuple(state_part_zoneout_prob * state_part + (1 - state_part_zoneout_prob) * new_state_part
                                  for new_state_part, state_part, state_part_zoneout_prob in
                                  zip(new_state, state, self._zoneout_prob))

            new_state = tf.cond(self.is_training, train, test)

        else:
            return NotImplemented

        new_state = nest.pack_sequence_as(structure=state, flat_sequence=new_state)

        return output, new_state