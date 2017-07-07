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