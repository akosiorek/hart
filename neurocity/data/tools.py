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