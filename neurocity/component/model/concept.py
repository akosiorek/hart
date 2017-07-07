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