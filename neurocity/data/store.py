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
import threading


class CircularValueStore(object):
    """Circcular value store, useful for iterating many times over objects, e.g.
    paths to imgs in a dataset. Supports shuffling every epoch.
    """

    def __init__(self, data_dict, num_epochs=None, shuffle=True):

        self.num_epochs = num_epochs
        self.shuffle = shuffle

        self.epoch = 0
        self._closed = False
        self.mutex = threading.Lock()
        self.reset_data(data_dict)

    def __len__(self):
        return len(self.values[0])

    def _generate_idx(self):
        if self.shuffle:
            return np.random.permutation(len(self))
        else:
            return np.arange(len(self))

    def get(self):
        self.mutex.acquire()
        try:
            if self.current_idx == len(self):
                if self.num_epochs is not None and self.epoch > self.num_epochs:
                    self._closed = True
                else:
                    self.current_idx = 0
                    self.idx = self._generate_idx()
                    self.epoch += 1

            if self.closed():
                raise ValueError('Store is closed')

            d = {k: v[self.idx[self.current_idx]] for k, v in zip(self.keys,
                                                                  self.values)}
            self.current_idx += 1
        finally:
            self.mutex.release()
        return d

    def close(self):
        self.mutex.acquire()
        self._closed = True
        self.mutex.release()

    def closed(self):
        return self._closed

    def reset_data(self, data_dict):
        self.mutex.acquire()

        self.keys = data_dict.keys()
        self.values = data_dict.values()

        # ensure equal num of entries in each value vec
        for v in self.values[1:]:
            assert len(v) == len(self)

        self.current_idx = len(self)
        self.idx = None

        self.mutex.release()