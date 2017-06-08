import os
import os.path as osp
import numpy as np
import scipy.sparse
import bt_net.config as cfg

class lmdb(object):
    """
    Language model database
    """

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name
