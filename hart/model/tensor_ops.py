import tensorflow as tf
from neurocity.tensor_ops import broadcast_against


def dynamic_truncate(v, t_max):
    """Truncates a tensor v with dynamic shape (T, ...) and
    static shape (None, ...) to (:t_max, ...) and preserves its
    static shape (useful if v is input to dynamic_rnn).

    :param v: time-major testing_data tensor
    :param t_max: scalar int32 tensor
    :return: truncated testing_data tensor
    """
    shape = list(v.get_shape())
    v = v[:t_max]
    v.set_shape(shape)
    return v


def select_present(x, presence, batch_size=1, name='select_present'):
    with tf.variable_scope(name):
        presence = 1 - tf.to_int32(presence)  # invert mask

        bs = x.get_shape()[0]
        if bs != None:  # here type(bs) is tf.Dimension and == is ok
            batch_size = int(bs)

        num_partitions = 2 * batch_size
        r = tf.range(0, num_partitions,  2)
        r.set_shape(tf.TensorShape(batch_size))
        r = broadcast_against(r, presence)

        presence += r

        selected = tf.dynamic_partition(x, presence, num_partitions)
        selected = tf.concat(axis=0, values=selected)
        selected = tf.reshape(selected, tf.shape(x))

    return selected


def inverse_selection(y, presence, batch_size=1, name='inverse_selection'):
    with tf.variable_scope(name):
        idx = tf.reshape(tf.range(tf.size(y)), tf.shape(y))
        idx = tf.reshape(select_present(idx, presence, batch_size), (-1,))
        idx = tf.invert_permutation(idx)

        x = tf.gather(tf.reshape(y, (-1,)), idx)
    return tf.reshape(x, tf.shape(y))


def normalize_contrast(x):
    """Normalize contrast of an image: forces values to be stricly in [0, 1]

    :param x: image tensor
    :return:
    """
    idx = tf.range(1, tf.rank(x))
    min = tf.reduce_min(x, idx, keep_dims=True)
    max = tf.reduce_max(x, idx, keep_dims=True)
    return (x - min) / (max - min + 1e-5)


def clip_interval(x1, w, X):
    """Clips the interval [x, x+w] to be within [0, X]"""
    x2 = x1 + w
    x = tf.minimum(x1, x2)
    x2 = tf.maximum(x1, x2)

    x_less_0 = tf.less(x, 0)
    w = tf.where(x_less_0, x2, w)
    x = tf.where(x_less_0, tf.zeros_like(x), x)

    x2_gt_X = tf.greater(x2, X)
    w = tf.where(x2_gt_X, X - x, w)
    w = tf.where(tf.greater(w, 0.), w, tf.zeros_like(w))
    x = tf.where(tf.greater(x, X), X * tf.ones_like(x), x)
    return x, w


def clip_bbox(bbox, img_size):
    """Clips the bbox=(y, x, h, w): [y, y+h]x[x, x+w] to be within [0, 0] x `img_size`"""
    with tf.variable_scope('clip_bbox'):
        axis = len(bbox.get_shape()) - 1
        y, x, h, w = tf.unstack(bbox, axis=axis)
        img_size = tf.to_float(tf.reshape(img_size, (-1,)))
        H, W = [tf.reshape(i, (1,)) for i in tf.unstack(img_size)[:2]]

        y, h = clip_interval(y, h, H)
        x, w = clip_interval(x, w, W)
        bbox = tf.stack((y, x, h, w), axis=axis)
    return bbox


def intersection_within(bbox, within):
    """Returns the coordinates of the intersection of `bbox` and `within`
    with respect to `within`

    :param bbox:
    :param within:
    :return:
    """
    x1 = tf.maximum(bbox[..., 1], within[..., 1])
    y1 = tf.maximum(bbox[..., 0], within[..., 0])
    x2 = tf.minimum(bbox[..., 1] + bbox[..., 3], within[..., 1] + within[..., 3])
    y2 = tf.minimum(bbox[..., 0] + bbox[..., 2], within[..., 0] + within[..., 2])
    w = x2 - x1
    w = tf.where(tf.less_equal(w, 0), tf.zeros_like(w), w)
    h = y2 - y1
    h = tf.where(tf.less_equal(h, 0), tf.zeros_like(h), h)

    y = y1 - within[..., 0]
    x = x1 - within[..., 1]

    area = h * w
    y = tf.where(tf.greater(area, 0.), y, tf.zeros_like(y))
    x = tf.where(tf.greater(area, 0.), x, tf.zeros_like(x))

    rank = len(bbox.get_shape()) - 1
    return tf.stack((y, x, h, w), rank)


def _bbox_to_mask(yy, region_size, dtype):
    # trim bounding box exeeding region_size on top and left
    neg_part = tf.nn.relu(-yy[:2])
    core = tf.ones(tf.to_int32(tf.round(yy[2:] - neg_part)), dtype=dtype)

    y1 = tf.maximum(yy[0], 0.)
    x1 = tf.maximum(yy[1], 0.)

    y2 = tf.minimum(region_size[0], yy[0] + yy[2])
    x2 = tf.minimum(region_size[1], yy[1] + yy[3])

    padding = (y1, region_size[0] - y2, x1, region_size[1] - x2)
    padding = tf.reshape(tf.stack(padding), (-1, 2))
    padding = tf.to_int32(tf.round(padding))
    mask = tf.pad(core, padding)

    # trim bounding box exeeding region_size on bottom and right
    rs = tf.to_int32(tf.round(region_size))
    mask = mask[:rs[0], :rs[1]]
    mask.set_shape((None, None))
    return mask


def _bbox_to_mask_fixed_size(yy, region_size, output_size, dtype):

    mask = _bbox_to_mask(yy, region_size, dtype)

    nonzero_region = tf.greater(tf.reduce_prod(tf.shape(mask)), 0)
    mask = tf.cond(nonzero_region, lambda: mask, lambda: tf.zeros(output_size, dtype))
    mask = tf.image.resize_images(mask[..., tf.newaxis], output_size)[..., 0]
    return mask


def bbox_to_mask(bbox, region_size, output_size, dtype=tf.float32):
    """Creates a binary mask of size `region_size` where rectangle given by
    `bbox` is filled with ones and the rest is zeros. Finally, the binary mask
    is resized to `output_size` with bilinear interpolation.

    :param bbox: tensor of shape (..., 4)
    :param region_size: tensor of shape (..., 2)
    :param output_size: 2-tuple of ints
    :param dtype: tf.dtype
    :return: a tensor of shape = (..., output_size)
    """
    shape = tf.concat(axis=0, values=(tf.shape(bbox)[:-1], output_size))
    bbox = tf.reshape(bbox, (-1, 4))
    region_size = tf.reshape(region_size, (-1, 2))

    def create_mask(args):
        yy, region_size = args
        return _bbox_to_mask_fixed_size(yy, region_size, output_size, dtype)

    mask = tf.map_fn(create_mask, (bbox, region_size), dtype=dtype)
    return tf.reshape(mask, shape)