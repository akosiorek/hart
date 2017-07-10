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
class Model(object):
    """Model should handle:
    * initialization of all variables
    * a separate graph (and a namespace) for itself
    * saving and loading parameters
    * exposing parameters for optimization
    """
    def __init__(self):

        self._init_input()
        self._init_exprs()
        self._init_output()


class SupervisedMixin(object):
    """(Un)supervised Mixins should:

    * initialize input placeholders
    * maybe setup variables for optimization"""
    def _init_input(self):
        self.input = ''
        self.target = ''


class UnsupervisedMixin(object):

    def _init_input(self):
        self.input = ''


class Autoencoder(Model, UnsupervisedMixin):
    """Model implementation should:

    * Use inputs initialized by a mixin
    * Implement the model
    * Handle initialization
    * It might need to configure some variables used by the base class
        and mixins"""

    def _init_exprs(self):
        pass