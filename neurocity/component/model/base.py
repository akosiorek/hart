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
import importlib
from collections import namedtuple


Modes = namedtuple('Modes', ['TRAIN', 'TEST'])
modes = Modes(TRAIN='TRAIN', TEST='TEST')


class BaseModel(object):
    """Superclass for TensorFlow-based Machine Learning models. It stores all layers
    and their variables and takes care of train/test modes.

    There is a global BaseModel by default and every Layer created by the user is added
    to the default model (unless created under a local model). Also every local model is
    added to the global model. It means that e.g. setting TRAIN mode on the global model
    can overwrite modes of local models."""

    def __init__(self):
        super(BaseModel, self).__init__()
        self.layers = []
        self.train_mode()

    def register(self, layer):
        Layer = importlib.import_module('neurocity.component.layer').Layer
        if not isinstance(layer, (BaseModel, Layer)):
            raise ValueError('{} is not derived from BaseModel or Layer classes'.format(layer))
        self.layers.append(layer)

    def train_mode(self, sess=None):
        for layer in self.layers:
            layer.train_mode(sess)
        self._mode = modes.TRAIN

    def test_mode(self, sess=None):
        for layer in self.layers:
            layer.test_mode(sess)
        self._mode = modes.TEST

    def mode(self):
        return self._mode


default_model = BaseModel()
_model_stack = [default_model]


def get_model():
    return _model_stack[-1]


def set_model(m):
    if not isinstance(m, BaseModel):
        raise ValueError('m should be of class {} but is {}'.format(BaseModel, m.__class__))
    global _model_stack
    _model_stack.append(m)


def reset_model():
    global _model_stack
    if len(_model_stack) > 1:
        _model_stack.pop()


class ModelHandle(object):
    def __enter__(self):
        return get_model()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # suppress if because of the lack of register method
        # if exc_type is AttributeError and exc_val.message.split()[-1] == "'register'":
        #     return True
        return exc_type is None


def train_mode(sess=None):
    """Sets the TRAIN mode globally.

    Note: It acts on the default model and all models created locally.

    :param sess: tf.Session"""

    get_model().train_mode(sess)


def test_mode(sess=None):
    """Sets the TEST mode globally.

    Note: It acts on the default model and all models created locally.

    :param sess: tf.Session"""

    get_model().test_mode(sess)


def mode():
    """Returns the global mode (TRAIN or TEST).

    Note: It can be different from modes of local models."""

    return get_model().mode()
