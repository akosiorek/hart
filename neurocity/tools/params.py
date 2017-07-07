import numpy as np
import tensorflow as tf


def num_trainable_params(graph=None):
    """Computes the total number of trainable parameters in the graph.

    :param graph: Tensorflow Graph; uses the default one if None
    :return: int
    """
    if graph is not None:
        vars = graph.trainable_variables()
    else:
        vars = tf.trainable_variables()

    n = 0
    for v in vars:
        n += np.prod([e.value for e in v.get_shape()])
    return n