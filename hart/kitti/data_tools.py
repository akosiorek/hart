import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest


def split_seq(seq, length, overlap=0, allow_shorter=True):
    """Splits a sequence into a list of sequences whose length is no greater
    than `length` and which overlap by `overlap` timesteps

    :param seq: a list-like object, a sequence
    :param length: int, max length of a sequence
    :param overlap: int, default 0
    :param allow_shorter: if True, the last of returned sequences might be shorter than `length`
    :return:
    """

    assert overlap < length
    if allow_shorter and len(seq) <= length:
        return [seq]

    parts = []
    a = 0
    b = length
    step = length - overlap
    while b <= len(seq):
        s = seq[a:b]
        if len(s) > 0:
            parts.append(s)
        a, b = a + step, b + step
    return parts


def split_seq_list(seq_list, length, overlap=0, allow_shorter=True):
    """Splits a list of sequences, see :func:`~split_seq`

    :param seq_list: list, a list of np.array objects
    """

    seqs = []
    for seq in seq_list:
        parts = split_seq(seq, length, overlap, allow_shorter)
        seqs.extend(parts)
    return seqs


def shuffle_together(*lists):
    """Shuffles all lists given in `lists` using the same permutation of indices,
    such that all entries at index `i` in lists[:][i, ...] before shuffling
    end up at index `k` lists[:][k, ...] after suffling.


    :param lists: list, a list of lists to shuffle
    :return: list, a list of shuffeled lists
    """
    assert len(lists) >= 1
    for l in lists[1:]:
        assert len(lists[0]) == len(l), 'Lenghts not equal!'

    index = np.random.permutation(len(lists[0]))
    outputs = [list() for _ in xrange(len(lists))]
    for i in xrange(len(outputs)):
        for j in index:
            outputs[i].append(lists[i][j])
    return outputs


def identity(x):
    return x


def run_py2tf_queue(py_queue_or_deque_op, dtypes, shapes=None, names=None, transform_fun=identity, n_threads=1, capacity=32,
                    dequeue_many=False, name='py2tf_queue', shuffle=False, enqueue_many=False, min_after_dequeue=None):
    """Converts a python Queue `py_queue` to a Tensorflow one by proessing entries from
     `py_queue` using `process_fun` and enqueuing them to a tf.FIFOQueue.

     The returned `dequeue_op` is a dictionary of ops with keys equal to `names` if `names`
     are provided and a list of ops otherwise.

     If shapes and/or names are provided, their lengths must match len(dtypes)

    :param py_queue_or_deque_op: python Queue object or a dequeue function
    :param dtypes: type or a list of types returned by process_fun
    :param shapes: int tuple or a list of int tuples, shapes returned by process_fun.
                Required for dequeue_many.
    :param names: string or a list of strings, names for the returned operations
    :param transform_fun: callable, function that processes entries in py_queue
    :param n_threads: int
    :param capacity: int
    :param dequeue_many: int, returns a minibatch of this size
    :param name: string, name of the variable scope
    :return: tensor dequeue op
    """

    assert dequeue_many is False or (dequeue_many > 1 and shapes is not None)
    if isinstance(dtypes, dict):
        names, dtypes = dtypes.keys(), dtypes.values()

    dtypes = nest.flatten(dtypes)
    if shapes is not None:
        shapes = nest.pack_sequence_as(dtypes, shapes)

    if names is not None:
        names = nest.flatten(names)

    assert all([isinstance(dt, (type, tf.DType)) for dt in dtypes]), \
        'Invalid dtypes, each type(dtype) must be in (type, tf.Dtype)'

    queue_shapes = shapes if shapes is not None and all(nest.flatten(shapes)) else None

    with tf.variable_scope(name):
        kwargs = dict(capacity=capacity, dtypes=dtypes, shapes=queue_shapes, names=names)

        if shuffle:
            if min_after_dequeue is None:
                min_after_dequeue = capacity // 2

            output_queue = tf.RandomShuffleQueue(min_after_dequeue=min_after_dequeue, **kwargs)
        else:
            output_queue = tf.FIFOQueue(**kwargs)

        py_deque_impl = getattr(py_queue_or_deque_op, 'get', py_queue_or_deque_op)

        def py_dequeue():
            values = transform_fun(py_deque_impl())
            if names is not None:
                values = [values[k] for k in names]
            return values

        tf_dequeue = tf.py_func(py_dequeue, [], dtypes, 'from_py_queue_to_tf')
        if names is not None:
            tf_dequeue = {k: v for k, v in zip(names, tf_dequeue)}

        enque_func = output_queue.enqueue_many if enqueue_many else output_queue.enqueue
        output_enqueue_op = enque_func(tf_dequeue)

        enqueue_ops = [output_enqueue_op] * n_threads
        queue_runner = tf.train.QueueRunner(output_queue, enqueue_ops)

        if dequeue_many:
            dequeue_op = output_queue.dequeue_many(dequeue_many)
        else:
            dequeue_op = output_queue.dequeue()

    tf.summary.scalar(
        '{}/fraction_of_{}_full'.format(name, capacity),
        tf.to_float(output_queue.size()) / capacity)

    # set shapes to dequeue ops
    if shapes is not None:
        if names is None:
            names = xrange(len(shapes))
        for k, s in zip(names, shapes):
            dequeue_op[k].set_shape(s)

    tf.train.add_queue_runner(queue_runner)
    return dequeue_op, output_queue.size()