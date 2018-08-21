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

import itertools
import tensorflow as tf
from tensorflow.python.ops import gen_math_ops

from neurocity.tools.params import num_trainable_params


def with_session(f):

    def wrapper(self, *args, **kwargs):
        sess = getattr(self, 'sess', None)
        if sess is not None:
            kwargs['sess'] = sess

        return f(self, *args, **kwargs)

    return wrapper


class TrainSchedule(object):

    def __init__(self, start_timesteps, max_timesteps, train_store=None, min_save_interval=1000,
                 min_logging_interval=0, max_logging_interval=None, log_every_n_epochs=1,
                 start_threshold=.7, max_threshold_decrease=0., min_epochs_per_timestep=None,
                 max_n_epochs=None, log_at_start=True):

        self.start_timesteps = start_timesteps
        self.max_timesteps = max_timesteps

        self.min_save_interval = min_save_interval
        self.min_logging_interval = min_logging_interval
        self.max_logging_interval = max_logging_interval
        self.log_every_n_epochs = log_every_n_epochs
        self.start_threshold = start_threshold
        self.max_threshold_decrease = max_threshold_decrease
        self.min_epochs_per_timestep = min_epochs_per_timestep
        self.max_n_epochs = max_n_epochs
        self.log_at_start = log_at_start

        self.n_timesteps = start_timesteps
        self.n_epoch = 0
        self.n_iter = 0
        self.n_iter_per_timestep = 0
        self.n_epochs_per_timestep = 0

        self.last_logged = 0
        self.sess = None

        self.train_store = train_store

    def increase_timesteps(self, sess=None):
        self.n_timesteps = min(self.n_timesteps + 1, self.max_timesteps)

    def time_increase_threshold(self):
        ratio = float(min(self.n_timesteps, self.max_timesteps) - self.start_timesteps) \
                / float(self.max_timesteps - self.start_timesteps)
        return self.start_threshold - ratio * self.max_threshold_decrease

    @with_session
    def report_score(self, score, sess=None):
        if score >= self.time_increase_threshold() and \
                (not self.min_epochs_per_timestep or self.n_epochs_per_timestep >= self.min_epochs_per_timestep):
            self.increase_timesteps(sess)
            self.n_epochs_per_timestep = self.n_iter_per_timestep = 0

    @property
    def train_store(self):
        return self._train_store

    @train_store.setter
    def train_store(self, store):
        self._train_store = store
        if self._train_store is not None:
            self._update_intervals()

    @property
    def n_batches_per_epoch(self):
        return self.train_store.n_batches_per_epoch

    @property
    def n_timesteps(self):
        return self._n_timesteps

    @n_timesteps.setter
    @with_session
    def n_timesteps(self, n_timesteps, sess=None):
        self._n_timesteps = n_timesteps
        try:
            self.train_store.set_length(self.n_timesteps, sess)
            self._update_intervals()

            print(self.train_stats())
        except AttributeError:
            pass

    @property
    def should_log(self):
        should = self.n_iter == 1 and self.log_at_start
        if should:
            self.log_at_start = False

        should = should or self.n_iter >= (self.last_logged + self.logging_interval)
        if should:
            self.last_logged = self.n_iter
        return should

    @property
    def should_save(self):
        return self.n_iter % self.save_interval == 0

    def _update_intervals(self):
            log_by_epochs = int(self.log_every_n_epochs * self.n_batches_per_epoch)
            if self.max_logging_interval is not None:
                log_by_epochs = min(self.max_logging_interval, log_by_epochs)

            self.logging_interval = max(self.min_logging_interval, log_by_epochs)
            self.save_interval = max(self.min_save_interval, self.logging_interval)

    def __iter__(self):
        print('Starting training at iter {}'.format(self.n_iter))
        print(self.train_stats())
        print('Num of trainable parameters:', num_trainable_params())

        for self.n_epoch in itertools.count(self.n_epoch + 1):
            self.n_epochs_per_timestep += 1
            if self.max_n_epochs and self.n_epoch > self.max_n_epochs:
                print('Finishing training after {} epochs'.format(self.max_n_epochs))
                break

            for self.n_iter in xrange(self.n_iter + 1, self.n_iter + self.n_batches_per_epoch + 1):
                self.n_iter_per_timestep += 1
                yield self.n_iter

    def train_stats(self):
        return 'n_timesteps = {}, train_batches = {}, logging_interval = {}' \
            .format(self.n_timesteps, self.train_store.n_batches_per_epoch, self.logging_interval)


class AdaptiveLoss(dict):
    name = 'loss'
    true_name = '/'.join((name, 'true'))

    def __init__(self, *args, **kwargs):
        super(AdaptiveLoss, self).__init__(*args, **kwargs)
        self[self.name] = self[self.true_name] = 0.
        self.weight_vars = dict()

    def add(self, name, l, weight=1., adaptive=True):
        # Book-keeping
        self.track(name, l)
        self[self.true_name] += weight * l

        # What's actually optimized
        weight = self.get_weight(name, weight, adaptive)
        self[self.name] += weight * l

    def get_weight(self, name, weight, adaptive):
        name = self._weight_name(name)
        if adaptive:
            weight_var = tf.get_variable(name, initializer=(tf.sqrt(weight)))
            weight = weight_var ** 2 + 1e-8
            tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, weight_var)
            self.weight_vars[name] = weight_var

        self[name] = tf.convert_to_tensor(weight)
        self['loss'] += tf.log(1 / weight)
        return weight

    def track(self, name, l):
        self['/'.join((self.name, name))] = l

    def assign_weights(self, weight_dict=None, **kwargs):

        if weight_dict is None:
            weight_dict = dict()

        weight_dict.update(kwargs)
        if not weight_dict:
            raise ValueError('Needs at least 1 weight to assign')

        assign_dict = dict()
        for k, v in weight_dict.iteritems():
            weight_name = self._weight_name(k)
            if weight_name in self.weight_vars:
                assign_dict[k] = self.weight_vars[weight_name].assign(v ** .5)
            else:
                print('Weight for loss "{}" is not adaptive. Skipping'.format(k))

        return assign_dict

    def _weight_name(self, name):
        return 'weight/{}'.format(name)

    @property
    def value(self):
        return self[self.name]

    @property
    def true_value(self):
        return self[self.true_name]


def minimize_clipped(optimizer, loss, clip_value, return_gvs=False, soft=False, **kwargs):
    """Computes a train_op with clipped gradients in the range [-clip_value, clip_value]

    :param optimizer: Tensorflow optimizer object
    :param loss: tensor
    :param clip_value: scalar value
    :param return_gvs: returns list of tuples of (gradient, parameter) for trainable variables
    :param kwargs: kwargs for optimizer.compute_gradients function
    :return: train_step
    """

    gvs = optimizer.compute_gradients(loss, **kwargs)
    clipped_gvs = [(g, v) for (g, v) in gvs if g is not None]

    if not soft:
        clipped_gvs = [(tf.clip_by_value(g, -clip_value, clip_value), v) for (g, v) in clipped_gvs]

    else:
        n_elems = 0
        norm_squared = 0.
        for g, v in gvs:
            n_elems += tf.reduce_prod(tf.shape(g))
            norm_squared += tf.reduce_sum(g ** 2)

        norm_squared /= tf.to_float(n_elems)
        inv_norm = gen_math_ops.rsqrt(norm_squared)
        cond = tf.greater(norm_squared, clip_value ** 2)

        def clip(x):
            return tf.cond(cond, lambda: clip_value * x * inv_norm, lambda: x)

        clipped_gvs = [(clip(g), v) for (g, v) in clipped_gvs]

    train_step = optimizer.apply_gradients(clipped_gvs)

    if return_gvs:
        train_step = (train_step, gvs)
    return train_step