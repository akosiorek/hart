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

################################################################################
# Code adapted by Adam Kosiorek from:
#
#
#
# Michael Guerzhoy and Davi Frossard, 2016
# AlexNet implementation in TensorFlow, with weights
# Details:
# http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
# With code from https://github.com/ethereon/caffe-tensorflow
# Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

import numpy as np
import tensorflow as tf


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(axis=3, num_or_size_splits=group, value=input)
        kernel_groups = tf.split(axis=3, num_or_size_splits=group, value=kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(axis=3, values=output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


alexnet_layers = ['conv4', 'conv5', 'maxpool5', 'fc6', 'fc7', 'probs']


def get_var(weights, name):
    idx = 0 if name.lower().endswith('w') else 1
    p = weights[name.split('_')[0]][idx]
    return tf.get_variable(name, initializer=p)


def alexnet(inpt, weights, until_layer=alexnet_layers[-1], maxpool=True):

    n_layers = 3
    if until_layer in alexnet_layers:
        n_layers += alexnet_layers.index(until_layer) + 1

    inpt -= tf.reduce_mean(inpt, (1, 2), keep_dims=True)

    # conv1
    # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11
    k_w = 11
    c_o = 96
    s_h = 4
    s_w = 4
    conv1W = get_var(weights, 'conv1_W')
    conv1b = get_var(weights, 'conv1_b')
    conv1_in = conv(inpt, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    # lrn1
    # lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2
    alpha = 2e-05
    beta = 0.75
    bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    if maxpool:
        # maxpool1
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        k_h = 3
        k_w = 3
        s_h = 2
        s_w = 2
        padding = 'VALID'
        lrn1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv2
    # conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5
    k_w = 5
    c_o = 256
    s_h = 1
    s_w = 1
    group = 2
    conv2W = get_var(weights, 'conv2_W')
    conv2b = get_var(weights, 'conv2_b')
    conv2_in = conv(lrn1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)

    # lrn2
    # lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2
    alpha = 2e-05
    beta = 0.75
    bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    if maxpool:
        # maxpool2
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
        k_h = 3
        k_w = 3
        s_h = 2
        s_w = 2
        padding = 'VALID'
        lrn2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv3
    # conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3
    k_w = 3
    c_o = 384
    s_h = 1
    s_w = 1
    group = 1
    conv3W = get_var(weights, 'conv3_W')
    conv3b = get_var(weights, 'conv3_b')
    conv3_in = conv(lrn2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    out = tf.nn.relu(conv3_in)

    if n_layers > 3:
        # conv4
        # conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3
        k_w = 3
        c_o = 384
        s_h = 1
        s_w = 1
        group = 2
        conv4W = get_var(weights, 'conv4_W')
        conv4b = get_var(weights, 'conv4_b')
        conv4_in = conv(out, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        out = tf.nn.relu(conv4_in)

    if n_layers > 4:
        # conv5
        # conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3
        k_w = 3
        c_o = 256
        s_h = 1
        s_w = 1
        group = 2
        conv5W = get_var(weights, 'conv5_W')
        conv5b = get_var(weights, 'conv5_b')
        conv5_in = conv(out, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        out = tf.nn.relu(conv5_in)

    if maxpool and n_layers > 5:
        # maxpool5
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        k_h = 3
        k_w = 3
        s_h = 2
        s_w = 2
        padding = 'VALID'
        out = tf.nn.max_pool(out, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    if n_layers > 6:
        fc6W = get_var(weights, 'fc6_W')
        fc6b = get_var(weights, 'fc6_b')
        out = tf.nn.relu_layer(tf.reshape(out, [-1, int(np.prod(out.get_shape()[1:]))]), fc6W, fc6b)

    if n_layers > 7:
        # fc7
        # fc(4096, name='fc7')
        fc7W = get_var(weights, 'fc7_W')
        fc7b = get_var(weights, 'fc7_b')
        out = tf.nn.relu_layer(out, fc7W, fc7b)

    if n_layers > 8:
        # fc8
        # fc(1000, relu=False, name='fc8')
        fc8W = get_var(weights, 'fc8_W')
        fc8b = get_var(weights, 'fc8_b')
        out = tf.nn.xw_plus_b(out, fc8W, fc8b)

        out = tf.nn.softmax(out)

    return out
