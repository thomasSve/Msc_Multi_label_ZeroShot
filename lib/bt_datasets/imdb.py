import os
import os.path as osp
import PIL
#from utils.cython_bbox import bbox_overlaps
import numpy as np
import scipy.sparse
import bt_net.config as cfg
import cPickle

class imdb(object):
    """ Image database """

    def __init__(self, name):
        self._name = name
        self._num_classes = 0
        self._classes = []
        self._image_index = []

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def num_images(self):
        return len(self._image_index)
        
        
