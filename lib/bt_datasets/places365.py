#!/usr/bin/env python
# Made by Bjoernar Remmen
# Short is better

import os, sys
import os.path as osp
from copy import deepcopy
from random import shuffle
from bt_datasets.imdb import imdb
from bt_net.config import cfg
import numpy as np
import scipy.io as sio
import cPickle
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from preprocessing.preprocess_images import read_img

from random import shuffle


class places365(imdb):
    def __init__(self, image_set):
        imdb.__init__(self, 'places365_' + image_set)
        self._image_set = image_set
        self._devkit_path = osp.join(cfg.IMAGE_DATA_DIR, 'places365')
        self._data_path = osp.join(self._devkit_path, 'data_256')


        self._classes = ('__background__',)
        with open(osp.join(self._devkit_path, 'labels.txt')) as lbls:
            for lbl_line in lbls:
                self._classes = self._classes + (lbl_line.rstrip(),)
        print "classes", len(self._classes)

        self._image_ext = ['.JPG']
        self._image_index = self._load_image_set_index()
        print "image_index", len(self._image_index)



        # Specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'top_k': 2000}

        assert os.path.exists(self._devkit_path), \
            'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)


    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt or self._data_path + /ImageSets/zero_shot.txt
        image_set_file = os.path.join(self._devkit_path, self._image_set + '.txt')
        print "img_file", image_set_file
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            image_index = []
            for line in f:
                line = line.split(",")
                image_index.append(line)
            #image_index = [line.split() for line in f]
        return image_index






    def get_image_data_clean(self, index, lang_db):
        data = self._image_index[index]
        image_data = data[0]
        lbls = [x.strip() for x in data[1:]]
        img_path = osp.join(self._data_path, image_data)
        word_vectors = []
        for lbl in lbls:
            word_vec = lang_db.word_vector(lbl)
            word_vectors.append(word_vec)
        return img_path , word_vectors


    def get_classes(self):
        return self._classes[1:]




