########################################################################################
# 
# Hierarchical Attentive Recurrent Tracking
# Copyright (C) 2017  Adam R. Kosiorek, Oxford Robotics Institute, University of Oxford
# email:   adamk@robots.ox.ac.uk
# webpage: http://ori.ox.ac.uk
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# 
########################################################################################

import collections

import tensorflow as tf
from tensorflow.python.util import nest
import numpy as np

from neurocity.component.model import base


def pad_shape(shape):
    return 1, shape[0], shape[1], 1


def ensure_square(shape):
    if not nest.is_sequence(shape):
        shape = (shape, shape)

    return shape


class Layer(object):
    """Abstract Class for base layer type
    * have a unique name
    * organize separate namespace for the layer variables
    * expose declared parameters/exressions as attributes
    * add summaries
    * add output expressions to tf.collection
    """
    counter = 0

    def __init__(self, inpt=None, name=None, weight_init=None, bias_init=None,
                 default_init=None):
        super(Layer, self).__init__()

        if inpt is not None:
            inpt = tf.convert_to_tensor(inpt)
        self.inpt = inpt

        if name is None:
            name = self._unique_name()
        self.name = name
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.default_init = default_init

        with tf.variable_scope(self.name, initializer=self.default_init) as vs:
            with base.ModelHandle() as model:
                self._forward(vs)
                model.register(self)

        self._mode = None
        self.output = tf.convert_to_tensor(self.output)

    def _unique_name(self):
        name = '{}{}'.format(self.__class__.__name__, Layer.counter)
        Layer.counter += 1
        return name

    def _forward(self, vs):
        raise NotImplementedError('Abstract Method')

    def _var(self, name, shape, dtype=None, init=None, trainable=True):
        if not isinstance(shape, collections.Sequence):
            shape = (shape,)

        name = '{}_{}'.format(name, self.name)
        return tf.get_variable(name, shape, dtype=dtype, initializer=init,
                               trainable=trainable)

    def train_mode(self, sess=None):
        self._mode = base.modes.TRAIN
        op = getattr(self, '_train_mode_op', None)
        if op is not None:
            sess.run(op)

    def test_mode(self, sess=None):
        self._mode = base.modes.TEST
        op = getattr(self, '_test_mode_op', None)
        if op is not None:
            sess.run(op)

    def mode(self):
        return self._mode

    def get_shape(self):
        return self.output.get_shape()


class AffineLayer(Layer):
    """Affine Layer, a.k.a. fully connected, inner product layer.

    This class also manages variable creation.
    """
    def __init__(self, inpt, n_output, transfer=tf.nn.relu, name=None,
                 weight_init=None, bias_init=None):

        self.n_output = n_output

        if transfer is None:
            transfer = lambda x: x
        self.transfer = transfer

        super(AffineLayer, self).__init__(inpt, name, weight_init, bias_init)

    def _forward(self, vs):
        self.n_inpt = self.inpt.get_shape()[-1]
        self.W = self._var('W', (self.n_inpt, self.n_output),
                           init=self.weight_init)
        self.b = self._var('b', self.n_output,
                           init=self.bias_init)

        self.pre = tf.matmul(self.inpt, self.W) + self.b
        self.output = self.transfer(self.pre)


class ConvLayer(Layer):
    """Conv2d Wrapper

    This class also manages variable creation.
    """
    def __init__(self, inpt, ksize, n_filters, strides=(1, 1), padding='SAME',
                 transfer=tf.nn.relu, name=None, weight_init=None, bias_init=None):

        self.ksize = ensure_square(ksize)
        self.n_filters = n_filters
        self.strides = ensure_square(strides)
        self.padding = padding

        if transfer is None:
            transfer = lambda x: x
        self.transfer = transfer

        super(ConvLayer, self).__init__(inpt, name, weight_init, bias_init)

    def _forward(self, vs):
        self.n_inpt = self.inpt.get_shape()[-1]
        shape = tuple(self.ksize) + (self.n_inpt, self.n_filters)
        self.W = self._var('W', shape,
                           init=self.weight_init)
        self.b = self._var('b', shape[-1],
                           init=self.bias_init)

        self.pre = tf.nn.conv2d(self.inpt, self.W, pad_shape(self.strides),
                                padding=self.padding) + self.b
        self.output = self.transfer(self.pre)


class MaxPoolLayer(Layer):
    """Max Pooling Wrapper"""
    def __init__(self, inpt, shape=(2, 2), strides=(2, 2),
                 padding='SAME', name=None):

        self.shape = ensure_square(shape)
        self.strides = ensure_square(strides)
        self.padding = padding

        super(MaxPoolLayer, self).__init__(inpt, name)

    def _forward(self, vs):
        self.output = tf.nn.max_pool(self.inpt, pad_shape(self.shape),
                                     pad_shape(self.strides), self.padding)


class DropoutLayer(Layer):
    """Dropout Wrapper"""
    def __init__(self, inpt, keep_prob, is_training=None, name=None):

        self.train_prob = keep_prob
        self.is_training = is_training

        super(DropoutLayer, self).__init__(inpt, name)

    def _forward(self, vs):
        if self.is_training is None:
            self.keep_prob = self._var('keep_prob', 1, init=tf.constant_initializer(self.train_prob),
                                       trainable=False)

            self._train_mode_op = self.keep_prob.assign((self.train_prob,))
            self._test_mode_op = self.keep_prob.assign((1.,))
            keep_prob = self.keep_prob[0]
        else:
            keep_prob = tf.where(self.is_training, self.train_prob, 1.0)

        self.output = tf.nn.dropout(self.inpt, keep_prob)


class BatchNormLayer(Layer):
    """Batch Norm Layer

    This layer is based on the following paper:
        Ioffe, Sergey, and Christian Szegedy.
        "Batch normalization: Accelerating deep network training by reducing
        internal covariate shift." arXiv preprint arXiv:1502.03167 (2015).
    """
    def __init__(self, inpt, is_training=None, name=None, **kwargs):

        self.bn_kwargs = kwargs
        self.bn_kwargs['is_training'] = is_training
        super(BatchNormLayer, self).__init__(inpt, name)

    def _forward(self, vs):
        if self.bn_kwargs['is_training'] is None:
            self.bn_kwargs['is_training'] = self._var('is_training', [], dtype=tf.bool,
                                                  init=tf.constant_initializer(True, tf.bool),
                                                  trainable=False)

            self._train_mode_op = self.bn_kwargs['is_training'].assign(True)
            self._test_mode_op = self.bn_kwargs['is_training'].assign(False)

        self.bn_kwargs['updates_collections'] = None
        self.output = tf.contrib.layers.batch_norm(self.inpt, **self.bn_kwargs)


class LayerNormLayer(Layer):
    """This layer is based on the following paper:
    Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton.
    "Layer normalization." arXiv preprint arXiv:1607.06450 (2016).
    """
    def __init__(self, inpt, transfer=None, name=None, eps=1e-8):

        self.transfer = transfer
        self.eps = eps
        super(LayerNormLayer, self).__init__(inpt, name)

    def _forward(self, vs):
        self.shift = self._var('shift', (1,), init=tf.constant_initializer())
        self.scale = self._var('scale', (1,), init=tf.constant_initializer(1))

        axes = range(1, len(self.inpt.get_shape()))
        mean, var = tf.nn.moments(self.inpt, axes, keep_dims=True)

        denom = tf.sqrt(tf.abs(var)) + self.eps
        self.output = self.scale * (self.inpt - mean) / denom + self.shift

        if self.transfer:
            self.output = self.transfer(self.output)


class ResizeLayer(Layer):
    """This layer upscales input (along W and H) with bilinear interpolation"""
    def __init__(self, inpt, size, name=None):
        if len(size) != 2:
            raise ValueError("Length of size should be 2.")
        self.size = size
        super(ResizeLayer, self).__init__(inpt, name)

    def _forward(self, vs):
        self.output = tf.image.resize_images(self.inpt, self.size)


class PadSpatialLayer(Layer):

    def __init__(self, inpt, pad_size, mode='REFLECT', name=None):

        pad_size = ensure_square(pad_size)
        self.paddings = [[0, 0], pad_size, pad_size, [0, 0]]
        self.pad_mode = mode
        super(PadSpatialLayer, self).__init__(inpt, name)

    def _forward(self, vs):
        self.output = tf.pad(self.inpt, self.paddings, self.pad_mode)


class DepthwiseConvLayer(Layer):
    """Depthwise Convolutional Layer

    This layer performs a per input channel conv2d operation. Note, this does
    not mix information across channels. This is usually followed with a 1x1
    convolution for the purpose of mixing channel information.

    Assuming k x k kernel shape and equal input and output channels (c) this has
    a complexity of:
    (c * m) * (k^2 + c)  instead of k^2 * c^2 for regular convolution.
    I.e. speedup = k*k*c/(m*(k*k+c))
    """
    def __init__(self, inpt, ksize, channel_multiplier, strides=(1, 1),
                 padding='SAME', transfer=tf.nn.relu, name=None,
                 weight_init=None, shared_filters=False):

        n_inpt = inpt.get_shape().as_list()[-1]

        ksize = ensure_square(ksize)
        strides = ensure_square(strides)

        if shared_filters:
            self.shape = tuple(ksize) + (1, channel_multiplier)
            self.n_rep = n_inpt
        else:
            self.shape = tuple(ksize) + (n_inpt, channel_multiplier)
            self.n_rep = 1

        self.strides = strides
        self.padding = padding

        if transfer is None:
            transfer = lambda x: x
        self.transfer = transfer

        super(DepthwiseConvLayer, self).__init__(inpt, name, weight_init)

    def _forward(self, vs):
        self.W = self._var('W', self.shape, init=self.weight_init)
        tiled_w = tf.tile(self.W, [1, 1, self.n_rep, 1])
        self.pre = tf.nn.depthwise_conv2d(self.inpt, tiled_w,
                                          pad_shape(self.strides),
                                          padding=self.padding)
        self.output = self.transfer(self.pre)


class AsymmetricConvLayer(Layer):
    """Asymmetric Convolutional 2D Layer.

    Assuming k x k kernel shape and equal input and output channels (c) this
    has a complexity of:
    2k * c^2  instead of k^2 * c^2 for regular convolution.
    Speedup = k/2
    """
    def __init__(self, inpt, ksize, n_filters, strides=(1, 1), padding='SAME',
                 transfer=tf.nn.relu, name=None, weight_init=None, bias_init=None,
                 bottleneck_filter_ratio=1):

        self.ksize = ensure_square(ksize)
        self.n_filters = n_filters
        self.strides = ensure_square(strides)
        self.padding = padding
        self.bottleneck_filter_ratio = bottleneck_filter_ratio

        if transfer is None:
            transfer = lambda x: x
        self.transfer = transfer

        super(AsymmetricConvLayer, self).__init__(inpt, name, weight_init,
                                                  bias_init)

    def _forward(self, vs):
        proj = ConvLayer(self.inpt, (1, self.ksize[1]),
                         int(self.n_filters/self.bottleneck_filter_ratio),
                         strides=(1, self.strides[1]),
                         padding=self.padding, transfer=None, name=self.name + "/x")
        self.output = ConvLayer(proj, (self.ksize[0], 1), self.n_filters,
                                strides=(self.strides[0], 1),
                                padding=self.padding,
                                transfer=self.transfer, name=self.name + "/y")


class StaticSpatialBiasLayer(Layer):
    """Static Spatial Bias Layer

    This layer learns a unique bias for each location in the input feature map.
    """
    def __init__(self, inpt, name=None, weight_init=None, per_channel=True):

        inpt_shape = inpt.get_shape()
        assert len(inpt_shape) == 4
        self.shape = inpt_shape
        self.shape[0] = 1
        if per_channel is False:
            self.shape[-1] = 1

        super(StaticSpatialBiasLayer, self).__init__(inpt, name, weight_init)

    def _forward(self, vs):
        self.W = self._var('W', self.shape,
                           init=self.weight_init)

        self.output = tf.add(self.inpt, self.W)


class DynamicFilterConvLayer(Layer):
    """Dynamic Filter Convolutional Layer

    This layer is based on the following paper:
        De Brabandere, Bert, et al. "Dynamic filter networks."
        Neural Information Processing Systems (NIPS). 2016.
    """
    def __init__(self, inpt, filters, ksize=1, n_filters=None,
                 strides=1, padding='SAME', name=None):
        """Interface for DynamicFilterConvLayer
        :param inpt:    A 4D tensor to apply filters to
        :param filters: A tensor which encodes the filters to be applied to the
                        inpt tensor. The shape of filters should be the
                        following: [B, H, W, F] where F is packed with filters
                         per location. I.e. F = (k_rows * k_cols * c_in * c_out)
        :param ksize: The spatial dimenions of the filters (k_rows, k_cols)
        :param n_filters: The number of output channels (c_out) corresponding to
                        the number of different filters flattened into the
                        'filters' param.
        :param strides: Same as conv2d
        :param padding: Same as conv2d
        :param name: Same as conv2d
        """

        inpt_shape = inpt.get_shape().as_list()
        filter_shape = filters.get_shape().as_list()
        assert len(inpt_shape) == 4
        assert len(filter_shape) == 4

        self.local = not filter_shape[1] == filter_shape[2] == 1
        self.ksize = ensure_square(ksize)
        self.n_cin = inpt_shape[-1]
        if n_filters is None:
            n_filters = np.round(filter_shape[-1] / (np.prod(self.ksize)*inpt_shape[-1])).astype(np.int32)

        self.strides = pad_shape(ensure_square(strides))
        assert filter_shape[-1] == self.n_cin * n_filters * np.prod(self.ksize)

        self.n_filters = n_filters
        self.padding = padding
        self.filters = filters

        super(DynamicFilterConvLayer, self).__init__(inpt, name)

    def _forward(self, vs):
        if self.local:  # expand input patches and split by filters
            input_local_expanded = tf.extract_image_patches(self.inpt,
                                                            pad_shape(self.ksize),
                                                            self.strides,
                                                            [1, 1, 1, 1],
                                                            padding=self.padding)

            values = []
            for filt in tf.split(axis=3, num_or_size_splits=self.n_filters, value=self.filters):
                channel_i = tf.reduce_sum(tf.multiply(filt, input_local_expanded), 3,
                                          keep_dims=True)
                values.append(channel_i)
            self.output = tf.concat(axis=3, values=values)
        else:  # split by images in batch and map to regular conv2d function
            inpt = tf.expand_dims(self.inpt, 1)

            filt_shape = [-1, self.ksize[0], self.ksize[1], self.n_cin, self.n_filters]
            filt = tf.reshape(self.filters, filt_shape)
            elems = (inpt, filt)
            result = tf.map_fn(lambda x: tf.nn.conv2d(x[0], x[1],
                                                      self.strides,
                                                      self.padding), elems,
                               dtype=tf.float32, infer_shape=False)
            result = tf.squeeze(result, [1])
            result.set_shape(self.inpt.get_shape()[:-1].concatenate([self.n_filters]))
            self.output = result


class IsTrainingLayer(Layer):

    def _forward(self, vs):
        self.is_training = self._var('is_training', 1, dtype=bool,
                  init=tf.constant_initializer(True), trainable=False)

        self._train_mode_op = self.is_training.assign((True,))
        self._test_mode_op = self.is_training.assign((False,))
        self.output = self.is_training[0]

