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