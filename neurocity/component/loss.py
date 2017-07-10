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

import inspect
import tensorflow as tf

from neurocity.tensor_ops import broadcast_against


def _points_per_mask_entry(data, mask):
    """Computes the number of elements in the data tensor masked by a single entry
    in the mask tensor

    :param data: data tensor
    :param mask: mask tensor
    :return: scalar tensor of type tf.float32
    """
    d, m = (tf.to_float(tf.reduce_prod(tf.shape(i))) for i in (data, mask))
    return d / m


def _mask(expr, mask):
    assert mask.dtype == tf.bool, '`mask`.dtype has to be tf.bool'
    mask_rank = tf.rank(mask)
    sample_shape = tf.shape(expr)[mask_rank:]
    flat_shape = tf.concat(([-1], sample_shape), 0)
    flat_expr = tf.reshape(expr, flat_shape)
    flat_mask = tf.reshape(mask, (-1,))

    return tf.boolean_mask(flat_expr, flat_mask)


def _mask_and_reduce(expr, mask):
    expr = _mask(expr, mask)
    return tf.reduce_mean(expr)


def _apply_mask_and_weights(expr, mask=None, weights=None, reduce=True):
    if weights is not None:
        expr *= tf.cast(broadcast_against(weights, expr), expr.dtype)

    if mask is not None:
        expr = _mask(expr, mask)

    if reduce:
        expr = tf.reduce_mean(expr)

    return expr


def wrap_mask_and_weights(loss_func):

    arg_spec = inspect.getargspec(loss_func)
    num_defaults = len(arg_spec.defaults) if arg_spec.defaults else 0
    num_required = len(arg_spec.args) - num_defaults
    if num_required == 2:
        def wrapper(a, b, mask=None, weights=None, reduce=True, **kwargs):
            l = loss_func(a, b, **kwargs)
            return _apply_mask_and_weights(l, mask, weights, reduce)
    elif num_required == 1:
        def wrapper(a, mask=None, weights=None, reduce=True, **kwargs):
            l = loss_func(a, **kwargs)
            return _apply_mask_and_weights(l, mask, weights, reduce)
    else:
        return NotImplemented

    return wrapper


@wrap_mask_and_weights
def intersection(a, b):
    x1 = tf.maximum(a[..., 1], b[..., 1])
    y1 = tf.maximum(a[..., 0], b[..., 0])
    x2 = tf.minimum(a[..., 1] + a[..., 3], b[..., 1] + b[..., 3])
    y2 = tf.minimum(a[..., 0] + a[..., 2], b[..., 0] + b[..., 2])
    w = x2 - x1
    w = tf.where(tf.less_equal(w, 0), tf.zeros_like(w), w)
    h = y2 - y1
    h = tf.where(tf.less_equal(h, 0), tf.zeros_like(h), h)
    return w * h


@wrap_mask_and_weights
def intersection_over_union(a, b):
    """Computes intersection over union for two tensors of bounding boxes given as
        a tensor of shape=(..., 4) where the last dimension is (y, x, h, w)

    :param a: tensor of shape (..., 4)
    :param b: tensor of shape (..., 4)
    :param mask: optional, boolean mask
    :return: scalar tensor
    """

    i_area = intersection(a, b, reduce=False)
    a_area = a[..., 2] * a[..., 3]
    b_area = b[..., 2] * b[..., 3]
    return i_area / (a_area + b_area - i_area)


@wrap_mask_and_weights
def mean_squared_error(x, y):
    """Computes the Mean Squared Error

    :param x: tensor
    :param y: tensor
    :param mask: boolean mask
    :return: scalar tensor
    """
    squared_diff = (x - y) ** 2
    return squared_diff


@wrap_mask_and_weights
def berhu(x, y, c=1.):
    """

    :param x: tensor
    :param y: tensor
    :param c: float indicating l1/l2 crossover point
    :param mask: boolean mask
    :return: tensor
    """
    abs_diff = tf.abs(x - y)
    squared_diff = tf.square(abs_diff)
    diff = tf.where(tf.greater(abs_diff, c), (squared_diff + c * c)/(2. * c), abs_diff)
    return diff


@wrap_mask_and_weights
def crossentropy(targets, pred, eps=1e-8):
    p = tf.maximum(pred, eps)
    n = tf.minimum(pred, 1. - eps)
    xe = targets * tf.log(p) + (1 - targets) * tf.log(1 - n)
    return -xe


@wrap_mask_and_weights
def negative_log_likelihood(expr, eps=1e-8):
    """Computes the negative log likelihood from probabilities given in `expr`.
    Makes sure that the result does not overflow.

    :param expr: tensor of probabilities
    :param eps:
    :return: tensor
    """
    return -tf.log(tf.where(tf.greater(expr, eps), expr, expr + eps))


class AdaptiveLoss(object):
    """Utility class for adaptive multi-objective loss adapted from a seminar
    given by Alex Kendall.

    Individual loss components are weighted by the uncertainty of the
    corresponding task. The total value of the loss differs from the true
    value due to additional regularization terms. The regularized value can be
    accessed via the `value` property. True true value is available as the
    `true_value` property.


    """
    name = 'loss'
    true_name = '/'.join((name, 'true'))
    weight_name = 'weight'

    def __init__(self, eps=1e-8, weight_summaries=False):
        """

        :param eps: float, a small value used to ensure numerical stability.
        :param weight_summaries: If True create summaries for adaptive weights.
        """
        super(AdaptiveLoss, self).__init__()

        self.eps = eps
        self.weight_summaries = weight_summaries

        self.values = dict()
        self.weights = dict()
        self.values[self.name] = self.values[self.true_name] = 0.

    def add(self, name, loss, weight=1., adaptive=True):
        """Adds `loss` to the overall loss and keeps track of the partial loss
        and its weight. If `adaptive` is True the `weight` is used to
        initialize a trainable tf.Variable and a regularization term is
        added to the overall loss.

        :param name: string, name of the loss
        :param loss: scalar tensor
        :param weight: scalar
        :param adaptive: boolean
        :return:
        """

        self.values[self.true_name] += weight * loss
        loss_name = '/'.join((self.name, name))
        self.values[loss_name] = loss

        weight_name = '/'.join((self.weight_name, name))
        weight = self.get_weight(name, weight, adaptive)
        self.weights[weight_name] = weight
        self.values[self.name] += weight * loss

        if self.weight_summaries:
            tf.summary.scalar("adaptive_loss/" + name, weight)

    def get_weight(self, name, weight, adaptive):
        """Initializes weight for a loss as a tf.Variable if it should
        be adaptive

        :param name: string, name of the weight variable
        :param weight: scalar, initial value for the weight
        :param adaptive: boolean
        :return:
        """
        if adaptive:
            weight = tf.get_variable(name, initializer=(weight ** .5 + self.eps)) ** 2
        self.values[self.name] += tf.log(1 / weight)
        return weight

    @property
    def value(self):
        return self.values[self.name]

    @property
    def true_value(self):
        return self.values[self.true_name]
