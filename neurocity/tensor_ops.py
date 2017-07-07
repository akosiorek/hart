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