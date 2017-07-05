import os
import pandas as pd
import numpy as np

from tensorflow.python.util import nest


def to_array_list(df, length=None, by_id=True):
    """Converts a dataframe to a list of arrays, with one array for every unique index entry.
    Index is assumed to be 0-based contiguous. If there is a missing index entry, an empty
    numpy array is returned for it.

    Elements in the arrays are sorted by their id.

    :param df:
    :param length:
    :return:
    """
    if by_id:
        assert 'id' in df.columns

        # if `id` is the only column, don't sort it (and don't remove it)
        if len(df.columns) == 1:
            by_id = False

    idx = df.index.unique()
    if length is None:
        length = max(idx) + 1

    l = [np.empty(0) for _ in xrange(length)]
    for i in idx:
        a = df.loc[i]
        if by_id:
            if isinstance(a, pd.Series):
                a = a[1:]
            else:
                a = a.copy().set_index('id').sort_index()

        l[i] = a.values.reshape((-1, a.shape[-1]))
    return np.asarray(l)


class KittiTrackingLabels(object):
    """Kitt Tracking Label parser. It can limit the maximum number of objects per track,
    filter out objects with class "DontCare", or retain only those objects present
    in a given frame.
    """

    columns = 'id class truncated occluded alpha x1 y1 x2 y2 xd yd zd x y z roty'.split()
    classes = 'Car Van Truck Pedestrian Person_sitting Cyclist Tram Misc DontCare'.split()


    def __init__(self, path_or_df, bbox_with_size=True, remove_dontcare=True, split_on_reappear=True,
                 truncated_threshold=2., occluded_threshold=3.):

        if isinstance(path_or_df, pd.DataFrame):
            self._df = path_or_df
        else:
            if not os.path.exists(path_or_df):
                raise ValueError('File {} doesn\'t exist'.format(path_or_df))

            self._df = pd.read_csv(path_or_df, sep=' ', header=None, names=self.columns,
                                   index_col=0, skip_blank_lines=True)

        self.bbox_with_size = bbox_with_size

        if remove_dontcare:
            self._df = self._df[self._df['class'] != 'DontCare']

        for c in self._df.columns:
            self._convert_type(c, np.float32, np.float64)
            self._convert_type(c, np.int32, np.int64)

        if not nest.is_sequence(occluded_threshold):
            occluded_threshold = (0, occluded_threshold)

        if not nest.is_sequence(truncated_threshold):
            truncated_threshold = (0, truncated_threshold)

        self._df = self._df[self._df['occluded'] >= occluded_threshold[0]]
        self._df = self._df[self._df['occluded'] <= occluded_threshold[1]]

        self._df = self._df[self._df['truncated'] >= truncated_threshold[0]]
        self._df = self._df[self._df['truncated'] <= truncated_threshold[1]]

        # make 0-based contiguous ids
        ids = self._df.id.unique()
        offset = max(ids) + 1
        id_map = {id: new_id for id, new_id in zip(ids, np.arange(offset, len(ids) + offset))}
        self._df.replace({'id': id_map}, inplace=True)
        self._df.id -= offset

        self.ids = list(self._df.id.unique())
        self.max_objects = len(self.ids)
        self.index = self._df.index.unique()

        if split_on_reappear:
            added_ids = self._split_on_reappear(self._df, self.presence, self.ids[-1])
            self.ids.extend(added_ids)
            self.max_objects += len(added_ids)

    def _convert_type(self, column, dest_type, only_from_type=None):
        cond = only_from_type is None or self._df[column].dtype == only_from_type
        if cond:
            self._df[column] = self._df[column].astype(dest_type)

    @property
    def bbox(self):
        bbox = self._df[['id', 'y1', 'x1', 'y2', 'x2']].copy()
        if self.bbox_with_size:
            bbox['y2'] -= bbox['y1']
            bbox['x2'] -= bbox['x1']

        """Converts a dataframe to a list of arrays

        :param df:
        :param length:
        :return:
        """

        return to_array_list(bbox)

    @property
    def presence(self):
        return self._presence(self._df, self.index, self.max_objects)

    @property
    def num_objects(self):
        ns = self._df.id.groupby(self._df.index).count()
        absent = list(set(range(len(self))) - set(self.index))
        other = pd.DataFrame([0] * len(absent), absent)
        ns = ns.append(other)
        ns.sort_index(inplace=True)
        return ns.as_matrix().squeeze()

    @property
    def cls(self):
        return to_array_list(self._df[['id', 'class']])

    @property
    def occlusion(self):
        return to_array_list(self._df[['id', 'occluded']])

    @property
    def id(self):
        return to_array_list(self._df['id'])

    def __len__(self):
        return self.index[-1] - self.index[0] + 1

    @classmethod
    def _presence(cls, df, index, n_objects):
        p = np.zeros((index[-1] + 1, n_objects), dtype=bool)
        for i, row in df.iterrows():
            p[i, row.id] = True
        return p

    @classmethod
    def _split_on_reappear(cls, df, p, id_offset):
        """Assign a new identity to an objects that appears after disappearing previously.
        Works on `df` in-place.

        :param df: data frame
        :param p: presence
        :param id_offset: offset added to new ids
        :return:
        """

        next_id = id_offset + 1
        added_ids = []
        nt = p.sum(0)
        start = np.argmax(p, 0)
        end = np.argmax(np.cumsum(p, 0), 0)
        diff = end - start + 1
        is_contiguous = np.equal(nt, diff)
        for id, contiguous in enumerate(is_contiguous):
            if not contiguous:

                to_change = df[df.id == id]
                index = to_change.index
                diff = index[1:] - index[:-1]
                where = np.where(np.greater(diff, 1))[0]
                for w in where:
                    to_change.loc[w + 1:, 'id'] = next_id
                    added_ids.append(next_id)
                    next_id += 1

                df[df.id == id] = to_change

        return added_ids


class KittiTrackingParser(object):

    def __init__(self, img_folder_or_paths=None, label_folder_or_labels=None,
                 bbox=True, id=True, cls=True, occlusion=True, presence=True,
                 truncated_threshold=2., occluded_threshold=3., mirror=False, reverse=False):

        self.data_dict = {}

        if img_folder_or_paths is not None:

            if isinstance(img_folder_or_paths, basestring):
                self.img_folder = img_folder_or_paths
                img_folder_or_paths = self._get_img_paths()

            self.data_dict['img_path'] = img_folder_or_paths

        label_parts = [bbox, id, cls, occlusion, presence]
        if label_folder_or_labels is not None and any(label_parts):

            if isinstance(label_folder_or_labels, basestring):
                self.label_folder = label_folder_or_labels
                label_folder_or_labels = self._get_labels(truncated_threshold, occluded_threshold)

            self.labels = label_folder_or_labels
            self.data_dict.update(self._extract_labels(label_parts))

            self._truncate_seqs()
            self._split_seqs()

    def _get_img_paths(self):
        assert os.path.isdir(self.img_folder)
        img_seq_names = []
        folders = []
        for seq_folder in os.listdir(self.img_folder):
            seq_folder = os.path.join(self.img_folder, seq_folder)
            if os.path.isdir(seq_folder):
                folders.append(seq_folder)

        folders = sorted(folders, key=lambda x: int(os.path.basename(x).split('.')[0]))
        for seq_folder in folders:
            img_names = [f for f in os.listdir(seq_folder) if f.endswith('.png')]
            img_names = sorted(img_names, key=lambda x: int(os.path.basename(x).split('.')[0]))
            img_names = [os.path.join(seq_folder, f) for f in img_names]
            img_seq_names.append(np.asarray(img_names))
        return img_seq_names

    def _get_labels(self, truncated_threshold, occluded_threshold):
        assert os.path.isdir(self.label_folder)
        labels = []
        label_files = [l for l in os.listdir(self.label_folder) if l.endswith('.txt')]
        label_files = sorted(label_files, key=lambda x: int(os.path.basename(x).split('.')[0]))
        for label_file in label_files:
              label_file = os.path.join(self.label_folder, label_file)
              labels.append(KittiTrackingLabels(label_file,
                                                truncated_threshold=truncated_threshold,
                                                occluded_threshold=occluded_threshold))
        return labels

    def _extract_labels(self, label_parts):
        data = {k: [] for k, v in zip(('bbox', 'id', 'cls', 'occlusion', 'presence'), label_parts) if v}
        for l in self.labels:
            for k, v in data.iteritems():
                v.append(getattr(l, k))
        return data

    def _truncate_seqs(self):
        """Remove images not present in label sequences"""
        for k, v in self.data_dict.iteritems():
            for i, label in enumerate(self.labels):
                if len(v[i]) != len(label.index):
                    self.data_dict[k][i] = v[i][label.index]

    def _split_seqs(self):
        """Splits sequences around parts for which no labels are present"""
        for i, label in enumerate(self.labels):
            diff = label.index[1:] - label.index[:-1]
            where = np.where(np.greater(diff, 1))[0]
            if len(where) != 0:
                where += 1
                starts, ends = [0] + list(where), list(where) + [max(label.index) + 1]

                for value_list in self.data_dict.values():
                    v = value_list[i]
                    for st, ed in zip(starts, ends):
                        value_list.append(v[st:ed])
                    del value_list[i]

    def __len__(self):
        return len(self.data_dict[self.data_dict.keys()[0]])
