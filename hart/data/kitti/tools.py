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
import os
import copy
from threading import Lock

import numpy as np
import tensorflow as tf

import neurocity as nct
from neurocity.data import store
from neurocity.data import tools as data_tools
from parser import KittiTrackingParser

try:
    from cv2 import imread, imresize
except ImportError:
    from scipy.misc import imread, imresize


def scatter(values, presence, choice=None, val_shape=None):
    """Scatters values in `values` according to the indicator vector `presence`.
    If `choice` is not None, retains only indices given by `choice`

    :param values: list of vectors
    :param presence: list of vectors
    :param choice: a single index vector
    :return: list of vectors
    """
    r = []
    shape = np.asarray(presence[0]).shape
    values, presence = ([np.asarray(i) for i in j] for j in (values, presence))

    if val_shape is None and values[0].ndim > len(shape):
        val_shape = max([np.asarray(v).shape[len(shape):] for v in values])

    if val_shape is not None:
        shape += tuple(val_shape)

    for y, p in zip(values, presence):
        z = np.zeros(shape)
        w = np.where(p)[0]
        if y.size > 0:
            z[w] = y
        if choice is not None:
            z = z[choice]

        r.append(z)
    return r


def split_sequence(seq, fraction=.5):
    assert (0. <= fraction <= 1.)

    fr = fraction if fraction >= .5 else 1 - fraction
    split = int(fr * len(seq) + 0.5)
    a, b = seq, None
    if len(seq) > 2:
        a, b = seq[:split], seq[split:]
        if len(b) == 1:
            a, b = seq, None

    if fraction < .5:
        a, b = b, a

    return a, b


def split_sequence_dict(data_dict, fraction=.5):
    """Splits sequences into sequences of length `fraction` and 1 - `fraction`

        `data_dict` is a dict of k: v pairs, where vs are lists of sequences.
        Every sequence in the list is split in two and put into two separate dicts.

        If any of the sequences in a pair has length of 1, then it is concatenated to the
         other one and put only in one of the dicts

    :param data_dict:
    :param fraction:
    :return: tuple of dicts with split sequences
    """
    a, b = dict(), dict()

    for k, v in data_dict.iteritems():
        a[k], b[k] = [], []
        for i in xrange(len(v)):
            sa, sb = split_sequence(v[i], fraction)
            if sa is not None:
                a[k].append(sa)
            if sb is not None:
                b[k].append(sb)

    return a, b


def shuffle_seq_dict(data):
    shuffeled = data_tools.shuffle_together(*data.values())
    for k, v in zip(data.keys(), shuffeled):
        data[k] = v
    return data


def resample_sequence_dict(data, max_len, overlap):
    """Splits sequences into shorter overlapping ones; Each value in dict `data` is a list
    of the same length and contains sequences. Matching sequences for different keys still match
    after resampling"""

    for k, v in data.iteritems():
        # TODO: fix 0-length splits in split_seq_list
        data[k] = data_tools.split_seq_list(data[k], max_len, overlap)
        for i in reversed(xrange(len(data[k]))):
            if len(data[k][i]) == 0:
                del data[k][i]
    return data


def sample_permutations(data, size, num_samples):
    """Creates `num_samples` permutations of size `size` of `data`

    :param data: list
    :param size: size of each returned list
    :param num_samples: number of returned permutations
    :return: list of permuted `data`
    """

    perms = {tuple(np.random.permutation(data.shape[0])[:size]) for _ in xrange(num_samples)}
    ws = [data[list(p)] for p in perms]
    return ws


def default_choice_fun(x, n):
    return [x[:n]]


def choose_n_objects(img_paths, bbox, presence, n_objects, choice_fun=default_choice_fun, *data):
    """Chooses up to `n_objects` objects"""
    total_objects = presence[0].sum()
    n = min(total_objects, n_objects)  # since we're basing of off 0th frame
    obj_indices = np.where(presence[0])[0]
    obj_indices = choice_fun(obj_indices, n)

    ip, b, p = outputs = [[] for _ in xrange(3)]
    # other_data = [[] for _ in xrange(len(data))]
    for obj_index in obj_indices:
        pp = presence[:, obj_index]

        # indices of objects present in the sequence for more than only a single timestep
        valid_length = np.where(np.not_equal(pp.sum(0), 1))[0]

        # truncate if no object there
        zeros = np.where(np.equal(pp.sum(-1), 0))[0]
        trunc_idx = zeros[0] if len(zeros) > 0 else None
        if trunc_idx == 1:
            continue  # skip length-1 seq

        ip.append(img_paths[:trunc_idx])

        scattered = scatter(bbox, presence, obj_index[valid_length])[:trunc_idx]
        b.append(scattered)
        p.append(pp[:trunc_idx, valid_length])

    return outputs


def sequences(data, max_len=None, shuffle=True, overlap=0, max_objects=None, sample_objects=False):
    """Generates sequences of length up to `max_len` (if given) overlapping by `overlap` timesteps.
    If `max_objects` is given, it limits per-sequence number of objects. If both `max_objects` is given
    and `sample_objects` evaluates to True, then:
        sample_objects >= 1: `sample_objects` permutations of length `max_objects` are sampled from objects
         in a given sequence.
        sample_objects < 0: (abs(sample_objects) * num-of-objects-in-a-sequence // max_objects) permutations of
         length `max_objects` are sampled from objects present in a given sequence.

    Each sequence starts with a maximum number of objects in that sequence. Objects can disappear before the end
    of the sequence.

    :param max_len: int, maximum length of the sequence
    :param shuffle: bool, shuffles sequences if True
    :param overlap: int, must be >= 0 and < max_len
    :param max_objects: int, maximum number of objects in a sequence
    :param sample_objects: bool or int, described above
    :return: dict of key: list o sequences
    """

    assert max_len is None or max_len > 0
    assert (max_len is None or overlap < max_len) and overlap >= 0
    assert max_objects is None or max_objects > 0

    if max_len is not None:
        data = resample_sequence_dict(data, max_len, overlap)

    if max_objects is not None:

        if sample_objects < 0:
            choice_fun = lambda x, n: sample_permutations(x, n,
                                                          max(1, abs(sample_objects) * (x.shape[0] // max_objects)))
        elif sample_objects >= 1:
            choice_fun = lambda x, n: [x[np.random.permutation(x.shape[0])[:n]] for _ in xrange(sample_objects)]
        else:
            choice_fun = default_choice_fun

        keys = ['img_path', 'bbox', 'presence']
        ip, b, p = (data[k] for k in keys)
        new_data = {k: [] for k in keys}
        for i in xrange(len(b)):
            chosen = choose_n_objects(ip[i], b[i], p[i], max_objects, choice_fun)
            for l, k in zip(chosen, keys):
                new_data[k].extend(l)

        data = new_data

    if shuffle:
        data = shuffle_seq_dict(data)
    return data


def read_img(path, size=None, dtype=np.float32):
    img = imread(path)
    if size is not None:
        img = imresize(img, size)

    return img.astype(dtype)


def read_imgs(arr, paths, img_store):
    try:
        for i, p in enumerate(paths):
            arr[i] = img_store[p]
    except Exception as e:
        print e
        raise


def pad_bbox(bboxes, num_objects):
    """Pad bboxes wih zeros along axis=1

    :param bboxes:
    :param num_objects:
    :return:
    """
    try:
        s = bboxes[0].shape[-1]
        bbox = np.zeros((len(bboxes), num_objects, s))
        for i, b in enumerate(bboxes):
            bbox[i, :b.shape[0]] = b
    except Exception as e:
        print e
        raise

    return bbox


def pad_to_size(d, size):
    """Pad to size along axis=-1

    :param d:
    :param size:
    :return:
    """
    pad_width = size - d.shape[-1]
    if pad_width > 0:
        padding = ((0, 0),) * (len(d.shape) - 1) + ((0, pad_width),)
        d = np.pad(d, padding, 'constant')

    return d


def process_entry(d, n_objects, img_store, depth_folder=None, bbox_scale=1.):
    paths = d['img_path']
    n_channels = 3 + int(depth_folder is not None)
    shape = np.concatenate(((len(paths),), img_store.img_size[:2], (n_channels,)))
    imgs = np.empty(shape, dtype=img_store.dtype)
    read_imgs(imgs[..., :3], paths, img_store)

    bbox = copy.deepcopy(d['bbox'])
    if bbox_scale != 1.:
        for i, b in enumerate(d['bbox']):
            bbox[i] = bbox_scale * b

    if depth_folder:
        paths = [os.path.join(depth_folder, '/'.join(p.split('/')[-2:])) for p in paths]
        read_imgs(imgs[..., 3:], paths, img_store)
    del d['img_path']

    if d['mirror']:
        for t in xrange(imgs.shape[0]):
            for c in xrange(imgs.shape[-1]):
                imgs[t, ..., c] = np.fliplr(imgs[t, ..., c])

        for i, b in enumerate(bbox):
            x = img_store.img_size[1] - b[..., 1] - b[..., 3]
            bbox[i][..., 1] = x

    del d['mirror']
    d['img'] = imgs
    d['bbox'] = pad_bbox(bbox, n_objects).astype(np.float32)
    d['presence'] = pad_to_size(d['presence'], n_objects).astype(np.uint8)
    return d


class ImageStore(dict):
    def __init__(self, img_size, in_memory=True, dtype=np.float32, **kwargs):

        super(ImageStore, self).__init__(**kwargs)
        self.img_size = img_size
        self.in_memory = in_memory

        if isinstance(dtype, tf.DType):
            dtype = getattr(np, dtype.name)

        self.dtype = dtype
        self.lock = Lock()

    def __getitem__(self, item):

        if self.in_memory:
            try:
                self.lock.acquire()
                if item not in self:
                    self[item] = read_img(item, self.img_size, self.dtype)
            except Exception as e:
                raise e
            finally:
                self.lock.release()
            return dict.__getitem__(self, item)
        else:
            return read_img(item, self.img_size)


class KittiStore(store.CircularValueStore):
    n_timesteps = None
    minibatch = None

    def __init__(self, data_dict, n_timesteps, img_size, batch_size, overlap_fraction=.5,
                 sample_objects=False, num_epochs=None, shuffle=True,
                 which_seqs=None, n_threads=3, in_memory=False, depth_folder=None,
                 storage_dtype=tf.float32, mirror=False, reverse=False, bbox_scale=1., name='',
                 deplete_queues_at_length_increase=True):

        assert isinstance(storage_dtype, tf.DType)

        self.data_dict = data_dict
        self.img_size = img_size
        self.batch_size = batch_size
        self.overlap_fraction = overlap_fraction
        self.sample_objects = sample_objects
        self.n_threads = n_threads
        self.in_memory = in_memory
        self.depth_folder = depth_folder
        self.storage_dtype = storage_dtype
        self.mirror = mirror
        self.reverse = reverse
        self.bbox_scale = bbox_scale
        self.name = name
        self.deplete_queues_at_length_increase = deplete_queues_at_length_increase

        if which_seqs is not None:
            self._filter_seqs(which_seqs)

        super(KittiStore, self).__init__(self.data_dict, num_epochs, shuffle)

        self.set_length(n_timesteps)

    def set_length(self, n_timesteps, sess=None):
        if self.n_timesteps != n_timesteps:

            overlap = int(self.overlap_fraction * n_timesteps)
            data = copy.deepcopy(self.data_dict)
            data = sequences(data, n_timesteps, False, overlap, 1,
                             sample_objects=self.sample_objects)

            data = self._mirror_reverse(data)
            data = shuffle_seq_dict(data)

            self.reset_data(data)
            self.n_batches_per_epoch = len(self) // self.batch_size

            # deplete the queues
            if self.deplete_queues_at_length_increase and self.n_timesteps is not None:
                while self.n_timesteps != n_timesteps:
                    self.n_timesteps = sess.run(self.minibatch.values())[0].shape[0]
            else:
                self.n_timesteps = n_timesteps

    def _filter_seqs(self, which_seqs):
        for k, v in self.data_dict.iteritems():
            self.data_dict[k] = [v[i] for i in which_seqs]

    def _mirror_reverse(self, data):

        # concat reversed sequences
        if self.reverse:
            for k, v in data.iteritems():
                reversed = [vv[::-1] for vv in v]
                data[k].extend(reversed)

        # add mirroring info
        n = len(data.values()[0])
        if not self.mirror:
            data['mirror'] = [False] * n
        else:
            for k, v in data.iteritems():
                data[k].extend(copy.deepcopy(v))
            data['mirror'] = [False] * n + [True] * n
        return data

    def get_minibatch(self):

        if self.minibatch is None:

            self.img_store = ImageStore(self.img_size, self.in_memory, self.storage_dtype)

            def get_single_sample():
                return process_entry(self.get(), 1, self.img_store,
                                     self.depth_folder, self.bbox_scale)

            n_channels = 3 + int(self.depth_folder is not None)
            shapes = [(None,) + tuple(self.img_size) + (n_channels,), (None, 1, 4),
                      (None, 1)]
            dtypes = [self.storage_dtype, tf.float32, tf.uint8]
            names = ['img', 'bbox', 'presence']

            sample, sample_queue_size = nct.run_py2tf_queue(get_single_sample, dtypes, shapes=shapes,
                                                            names=names, n_threads=self.n_threads,
                                                            capacity=2 * self.batch_size,
                                                            name='{}/py2tf_queue'.format(self.name))

            minibatch = tf.train.batch(sample, self.batch_size, dynamic_pad=True, capacity=2)

            for k, v in minibatch.iteritems():
                unpacked = tf.unstack(v)
                unpacked = [u[:, tf.newaxis] for u in unpacked]
                minibatch[k] = tf.concat(axis=1, values=unpacked)

            if self.storage_dtype != tf.float32:
                minibatch[names[0]] = tf.to_float(minibatch[names[0]])
                dtypes[0] = tf.float32

            queue = tf.FIFOQueue(2, dtypes, names=names)
            enqeue_op = queue.enqueue(minibatch)
            runner = tf.train.QueueRunner(queue, [enqeue_op] * 2)
            tf.train.add_queue_runner(runner)
            minibatch = queue.dequeue()
            for name, shape in zip(names, shapes):
                minibatch[name].set_shape((shape[0], self.batch_size) + shape[1:])

            self.minibatch = minibatch

        return self.minibatch


def get_data(img_folder, label_folder, train_fraction, img_size,
             train_timesteps=4, test_timesteps=4, batch_size=1, sample_objects=False, n_threads=3,
             in_memory=False, which_seqs=None, truncated_threshold=2., occluded_threshold=3., depth_folder=None,
             storage_dtype=tf.uint8, mirror=False, reverse=False, bbox_scale=.5):
    kitti = KittiTrackingParser(img_folder, label_folder, presence=True, id=False, cls=False,
                                truncated_threshold=truncated_threshold, occluded_threshold=occluded_threshold)

    train, test = split_sequence_dict(kitti.data_dict, train_fraction)

    def make_store(name, d, timesteps, n_threads, mirror=False, reverse=False):
        s = KittiStore(d, timesteps, img_size, batch_size,
                       sample_objects=sample_objects, which_seqs=which_seqs, n_threads=n_threads,
                       in_memory=in_memory, depth_folder=depth_folder, storage_dtype=storage_dtype,
                       mirror=mirror, reverse=reverse, bbox_scale=bbox_scale, name=name)
        return s

    train_store = make_store('train', train, train_timesteps, n_threads, mirror, reverse)
    test_store = make_store('test', test, test_timesteps, (n_threads // 2) + 1)

    return train_store, train_store.get_minibatch(), test_store, test_store.get_minibatch()