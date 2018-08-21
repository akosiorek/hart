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

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

from hart.model.alexnet import alexnet
from neurocity.component.model import Model
from neurocity.tensor_ops import convert_shape
from neurocity.component.layer import (ensure_square, Layer, AffineLayer, ConvLayer, DropoutLayer, LayerNormLayer,
                                       BatchNormLayer, DynamicFilterConvLayer)


def normalise_activations(inpt, type=None, transfer=tf.nn.relu, is_training=None, name=None):
    if name is not None and type is not None:
        name = '{}_{}_norm'.format(name, type)
    if type == 'layer':
        inpt = LayerNormLayer(inpt, name=name)
    elif type == 'batch':
        inpt = BatchNormLayer(inpt, decay=.95, scale=True, is_training=is_training, name=name)
    elif type:
        raise ValueError('Invalid type: {}'.format(type))

    if transfer:
        inpt = transfer(inpt)
    return inpt


class IsTrainingLayer(Layer):

    def _forward(self, vs):
        self.is_training = self._var('is_training', 1, dtype=bool,
                  init=tf.constant_initializer(True), trainable=False)

        self._train_mode_op = self.is_training.assign((True,))
        self._test_mode_op = self.is_training.assign((False,))
        self.output = self.is_training[0]


class DynamicFilterModel(Model):
    def __init__(self, inpt, filter_inpt, ksize, n_channels, n_param_layers=2, transfer=tf.nn.elu, adaptive_bias=True,
                 bias=True, dfn_weight_factor=.1, dfn_bias_factor=2. ** .5, name=None):

        assert n_param_layers >= 1

        self.inpt = inpt
        self.filter_inpt = filter_inpt
        self.ksize = ensure_square(ksize)
        self.n_channels = n_channels
        self.n_param_layers = n_param_layers
        self.transfer = transfer
        self.adaptive_bias = adaptive_bias
        self.bias = bias
        self.dfn_weight_factor = dfn_weight_factor
        self.dfn_bias_factor = dfn_bias_factor

        name = self.__class__.__name__ if name is None else name
        super(DynamicFilterModel, self).__init__(name=name)

    def _build(self):
        n_inpt_channels = self.inpt.get_shape().as_list()[-1]
        n_dfn_filter_params = n_inpt_channels * self.n_channels * np.prod(self.ksize)

        filter_inpt = self.filter_inpt
        for i in range(1, self.n_param_layers):
            filter_inpt = AffineLayer(filter_inpt, filter_inpt.get_shape().as_list()[-1],
                                      transfer=tf.nn.elu, name='param_layer_{}'.format(i))

        dfn_weight_init = tf.uniform_unit_scaling_initializer(self.dfn_weight_factor)
        self.dynamic_weights = AffineLayer(filter_inpt, n_dfn_filter_params, transfer=None,
                                           weight_init=dfn_weight_init, bias_init=dfn_weight_init, name='dynamic_weights')

        dfn_weights = tf.reshape(self.dynamic_weights, (-1, 1, 1, n_dfn_filter_params))
        dfn = DynamicFilterConvLayer(self.inpt, dfn_weights, self.ksize, name='dfn')

        if self.adaptive_bias:
            dfn_bias_init = tf.uniform_unit_scaling_initializer(self.dfn_bias_factor)
            self.dynamic_bias = AffineLayer(filter_inpt, self.n_channels, transfer=None,
                                            weight_init=dfn_bias_init, bias_init=dfn_bias_init,
                                            name='dynamic_bias')

            dfn_adaptive_bias = tf.reshape(self.dynamic_bias, (-1, 1, 1, self.n_channels))
            dfn += dfn_adaptive_bias

        if self.bias:
            self.bias = tf.get_variable('dfn_bias', (1, 1, 1, self.n_channels))
            dfn += self.bias

        self.features = self.transfer(dfn)


class FeatureExtractor(Model):
    vs = None

    def __call__(self, inpt, vs=None, reuse=False):
        if self.vs is None:
            if vs is None:
                vs = tf.get_variable_scope()
            self.vs = vs

        self.inpt = inpt
        with tf.variable_scope(self.vs, reuse=reuse):
            super(FeatureExtractor, self).__init__(self.__class__.__name__)

        return self.features


class AlexNetModel(FeatureExtractor):
    def __init__(self, model_folder, n_out_feature_maps=10, layer='maxpool5', upsample=True, normlayer=None,
                 keep_prob=None, is_training=True):

        self.n_out_feature_maps = n_out_feature_maps
        self.path = os.path.join(model_folder, 'bvlc_alexnet.npy')
        self.layer = layer
        self.upsample = upsample
        self.normlayer = normlayer
        self.keep_prob = keep_prob
        self.is_training = is_training
        self.vs = None

    def _build(self):

        inpt = self.inpt
        if self.upsample:
            inpt = tf.image.resize_bilinear(inpt, (227, 227))
        o = alexnet(inpt, np.load(self.path, encoding = 'latin1').item(), self.layer, maxpool=self.upsample)
        self.orig_output = o

        if 'maxpool' or 'conv' in self.layer:
            o = ConvLayer(o, (1, 1), self.n_out_feature_maps, transfer=None, name='readout')
            self.readout = o

        if self.keep_prob:
            o = DropoutLayer(o, self.keep_prob, self.is_training, name='dropout').output

        self.features = normalise_activations(o, self.normlayer, tf.nn.elu, self.is_training, name='bn')
        self.n_features = convert_shape(self.features.get_shape()[1:], np.int32)


class MLP(Layer):
    """Multilayer Perceptron"""

    def __init__(self, inpt, n_hidden, n_output, transfer_hidden=tf.nn.elu, transfer=None,
                 hidden_weight_init=None, hidden_bias_init=None,weight_init=None, bias_init=None,
                 name=None):
        """
        :param inpt: inpt tensor
        :param n_hidden: scalar ot list, number of hidden units
        :param n_output: scalar, number of output units
        :param transfer_hidden: scalar or list, transfers for hidden units. If list, len must be == len(n_hidden).
        :param transfer: tf.Op or None
        """

        self.n_hidden = nest.flatten(n_hidden)
        self.n_output = n_output
        self.hidden_weight_init = hidden_weight_init
        self.hidden_bias_init = hidden_bias_init

        transfer_hidden = nest.flatten(transfer_hidden)
        if len(transfer_hidden) == 1:
            transfer_hidden *= len(self.n_hidden)
        self.transfer_hidden = transfer_hidden

        self.transfer = transfer
        super(MLP, self).__init__(inpt, name, weight_init, bias_init)

    def _forward(self, vs):
        inpt = self.inpt
        i = 0

        for i, (n_units, transfer) in enumerate(zip(self.n_hidden, self.transfer_hidden)):
            inpt = AffineLayer(inpt, n_units, transfer, 'layer_{}'.format(i),
                               self.hidden_weight_init, self.hidden_bias_init)

        self.output = AffineLayer(inpt, self.n_output, self.transfer, 'layer_{}'.format(i + 1),
                               self.weight_init, self.bias_init)
