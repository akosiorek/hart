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

import threading
import weakref
import queue

import tensorflow as tf
from tensorflow.python.util import nest


tf.GraphKeys.NCT_QUEUE_RUNNERS = "nct_queue_runners"


def start_queue_runners(sess=None, coord=None, daemon=True, start=True,
                        collection=tf.GraphKeys.NCT_QUEUE_RUNNERS):
  """Starts all queue runners collected in the graph, along with python
  queue runners.

  Args:
    sess: `Session` used to run the queue ops.  Defaults to the
      default session.
    coord: Optional `Coordinator` for coordinating the started threads.
    daemon: Whether the threads should be marked as `daemons`, meaning
      they don't block program exit.
    start: Set to `False` to only create the threads, not start them.
    collection: A `GraphKey` specifying the graph collection to
      get the queue runners from.  Defaults to `GraphKeys.QUEUE_RUNNERS`.

  Returns:
    A list of threads.
  """
  threads = tf.train.start_queue_runners(sess, coord, daemon, start)
  for qr in tf.get_collection(collection):
      threads.extend(qr.create_threads(sess, coord, daemon, start))

  return threads


class PyQueueRunner(object):
    """Holds a list of enqueue operations for a python queue, each to be run in a thread.

    Uses tf.train.Coordinator to manage threads; useful for creating data pipelines.
    """

    def __init__(self, queue, enqueue_ops):
        """Create a PyQueueRunner.

        When you later call the `create_threads()` method, the `QueueRunner` will
        create one thread for each op in `enqueue_ops`.  Each thread will run its
        enqueue op in parallel with the other threads.  The enqueue ops do not have
        to all be the same op, but it is expected that they all enqueue tensors in
        `queue`.

        Args:
          qnqueue_handler: a python function that transforms
          queue: A `Queue`.
          enqueue_ops: List of enqueue ops to run in threads later.
        """
        self._queue = queue
        self._enqueue_ops = enqueue_ops

        self._lock = threading.Lock()
        # A map from a session object to the number of outstanding queue runner
        # threads for that session.
        self._runs_per_session = weakref.WeakKeyDictionary()

    @property
    def queue(self):
        return self._queue

    @property
    def enqueue_ops(self):
        return self._enqueue_ops

    # pylint: disable=broad-except
    def _run(self, sess, enqueue_op, coord=None):
        """Execute the enqueue op in a loop.

        Args:
          sess: A Session; only for compatiblity with tf.train.Coordinator.
          enqueue_op: The python function to run. It's output is enqueued.
          coord: Coordinator object for reporting errors and checking
            for stop conditions.
        """
        try:
            while True:
                if coord and coord.should_stop():
                    break

                enqueue_op()
        except Exception as e:
            if coord:
                coord.request_stop(e)
                tf.logging.error("Exception in QueueRunner: %s", str(e))
        finally:
            with self._lock:
                self._runs_per_session[sess] -= 1

    def create_threads(self, sess, coord, daemon=False, start=False):
        """Create threads to run the enqueue ops for the given session.

        This method requires a session in which the graph was launched.  It creates
        a list of threads, optionally starting them.  There is one thread for each
        op passed in `enqueue_ops`.

        The `coord` argument is a coordinator that the threads will use
        to terminate together and report exceptions.

        If previously created threads for the given session are still running, no
        new threads will be created.

        Args:
          sess: A `Session`.
          coord: Optional `Coordinator` object for reporting errors and checking
            stop conditions.
          daemon: Boolean.  If `True` make the threads daemon threads.
          start: Boolean.  If `True` starts the threads.  If `False` the
            caller must call the `start()` method of the returned threads.

        Returns:
          A list of threads.
        """
        with self._lock:
            try:
                if self._runs_per_session[sess] > 0:
                    # Already started: no new threads to return.
                    return []
            except KeyError:
                # We haven't seen this session yet.
                pass
            self._runs_per_session[sess] = len(self._enqueue_ops)

        ret_threads = [threading.Thread(target=self._run, args=(sess, op, coord))
                       for op in self._enqueue_ops]

        for t in ret_threads:
            if coord:
                coord.register_thread(t)
            if daemon:
                t.daemon = True
            if start:
                t.start()
        return ret_threads


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


def run_py2py_queue(py_queue=None, transform_fun=identity, n_threads=1,
                    capacity=32, dequeue_op=None, enqueue_many=False):
    """Applies `process_fun` on elements taken from `py_queue` or returned by `dequeue_op`
    and stores the results in a new python Queue

    :param py_queue: python Queue object, must have `get` method
    :param transform_fun: function applied to elements
    :param n_threads: int, number of processing threads
    :param capacity: int, max size of the output queue
    :param dequeue_op: function, should be specified only if `py_queue` isn't
    :return: output queue
    """
    # one of py_queue or deqeueue_op must be defined, but both can't
    assert py_queue is None or dequeue_op is None
    assert py_queue is not None or dequeue_op is not None

    queue = Queue.Queue(capacity)
    queue.name = 'py_fifo_queue'

    if dequeue_op is None:
        dequeue_op = py_queue.get

    def py2py_transform_op():
        return transform_fun(dequeue_op())

    if enqueue_many:
        def enqueue_op():
            for element in py2py_transform_op():
                queue.put(element)
    else:
        def enqueue_op():
            queue.put(py2py_transform_op())

    runner = PyQueueRunner(queue, [enqueue_op] * n_threads)
    tf.add_to_collection(tf.GraphKeys.NCT_QUEUE_RUNNERS, runner)
    return queue
