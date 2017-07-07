import time
import numpy as np
import tensorflow as tf

from tensorflow.python.util import nest


def make_expr_logger(sess, writer, num_batches, expr_dict, name, data_dict=None,
                     constants_dict=None, measure_time=True):
    """

    :param sess:
    :param writer:
    :param num_batches:
    :param expr:
    :param name:
    :param data_dict:
    :param constants_dict:
    :return:
    """

    tags = {k: '/'.join((k, name)) for k in expr_dict}
    data_name = 'Data {}'.format(name)
    log_string = ', '.join((''.join((k + ' = {', k, ':.4f}')) for k in expr_dict))
    log_string = ' '.join(('Step {},', data_name, log_string))

    if measure_time:
        log_string += ', eval time = {:.4}s'

        def log(itr, l, t): return log_string.format(itr, t, **l)
    else:
        def log(itr, l, t): return log_string.format(itr, **l)

    def logger(itr=0, num_batches_to_eval=None, write=True):
        l = {k: 0. for k in expr_dict}
        start = time.time()
        if num_batches_to_eval is None:
            num_batches_to_eval = num_batches

        for i in xrange(num_batches_to_eval):
            if data_dict is not None:
                vals = sess.run(data_dict.values())
                feed_dict = {k: v for k, v in zip(data_dict.keys(), vals)}
                if constants_dict:
                    feed_dict.update(constants_dict)
            else:
                feed_dict = constants_dict

            r = sess.run(expr_dict, feed_dict)
            for k, v in r.iteritems():
                l[k] += v
        
        for k, v in l.iteritems():
            l[k] /= num_batches_to_eval
        t = time.time() - start
        print log(itr, l, t)

        if write:
            log_values(writer, itr, [tags[k] for k in l.keys()], l.values())

        return l

    return logger


def log_ratios(var_tuples, name='ratio', which_name=1, eps=1e-8):
    """

    :param var_tuples:
    :param name:
    :param which_name:
    :param eps:
    :return:
    """
    for tup in var_tuples:
        op_name = '/'.join((tup[which_name].name, name))
        a, b = tup
        ratio = tf.reduce_mean(abs(a) / (abs(b) + eps))
        tf.summary.scalar(op_name, ratio)


def log_norm(expr_list, name):
    """

    :param expr_list:
    :param name:
    :return:
    """
    n_elems = 0
    norm = 0.
    for e in expr_list:
        n_elems += tf.reduce_prod(tf.shape(e))
        norm += tf.reduce_sum(e**2)
    norm /= tf.to_float(n_elems)
    tf.summary.scalar(name, norm)
    return norm


def log_values(writer, itr, tags=None, values=None, dict=None):

    if dict is not None:
        assert tags is None and values is None
        tags = dict.keys()
        values = dict.values()
    else:

        if not nest.is_sequence(tags):
            tags, values = [tags], [values]

        elif len(tags) != len(values):
            raise ValueError('tag and value have different lenghts:'
                             ' {} vs {}'.format(len(tags), len(values)))

    for t, v in zip(tags, values):
        summary = tf.Summary.Value(tag=t, simple_value=v)
        summary = tf.Summary(value=[summary])
        writer.add_summary(summary, itr)


def check_id_swaps(bboxes, presence, factor=1.):
    for i in xrange(bboxes.shape[1]):
        db = abs(bboxes[1:, i] - bboxes[:-1, i])
        p = presence[1:, i]
        counts = p.sum(0)[:, np.newaxis] + 1e-8
        m = db.sum(0) / counts
        db -= m
        std = np.sqrt(((db - m) ** 2).sum(0) / counts)

        margin = factor * std
        errors = np.where(np.greater(np.greater(db, margin).sum(-1).max(-1), 2))[0]
        if len(errors) > 0:
            print 'Possible id swaps at frame(s) {} in seq {}'.format(errors + 1, i)


def image_series_summary(tag, imgs, max_timesteps=10):
    # take only 3 items from the minibatch
    imgs = imgs[:, :3]

    # assume img.shape == (T, batch_size, n_obj, H, W, C)
    # let's log only for 1st obj
    tf.cond(tf.equal(tf.rank(imgs), 6), lambda: imgs[:, :, 0], lambda: imgs)

    shape = (max_timesteps,) + tuple(imgs.get_shape()[1:])
    nt = tf.shape(imgs)[0]

    def pad():
        paddings = tf.concat(axis=0, values=([[0, max_timesteps - nt]], tf.zeros((len(shape) - 1, 2), tf.int32)))
        return tf.pad(imgs, paddings)

    imgs = tf.cond(tf.greater(nt, max_timesteps), lambda: imgs[:max_timesteps], pad)
    imgs.set_shape(shape)
    imgs = tf.squeeze(imgs)
    imgs = tf.unstack(imgs)

    # concatenate along the columns
    imgs = tf.concat(axis=2, values=imgs)
    tf.summary.image(tag, imgs)


