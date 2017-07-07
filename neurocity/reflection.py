import collections


def is_numpy_type(x):
    return x.__class__.__module__ == 'numpy'