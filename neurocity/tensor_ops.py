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

import numpy as np
import tensorflow as tf


def convert_shape(shape, dtype=np.int32):
    """Converts `shape` of type tf.TensorShape to a np.array of numbers

    :param shape: tf.TensorShape
    :param dtype: output dtype
    :return: np.array
    """
    if not isinstance(shape, tf.TensorShape):
        return shape

    ss = map(lambda x: x if x is not None else -1, shape.as_list())

    return np.asarray(ss, dtype=dtype)


def broadcast_against(tensor, against_expr):
    """Adds trailing dimensions to mask to enable broadcasting against data

    :param tensor: tensor to be broadcasted
    :param against_expr: tensor will be broadcasted against it
    :return: mask expr with tf.rank(mask) == tf.rank(data)
    """

    def cond(data, tensor):
        return tf.less(tf.rank(tensor), tf.rank(data))

    def body(data, tensor):
        return data, tf.expand_dims(tensor, -1)

    shape_invariants = [against_expr.get_shape(), tf.TensorShape(None)]
    _, tensor = tf.while_loop(cond, body, [against_expr, tensor], shape_invariants)
    return tensor