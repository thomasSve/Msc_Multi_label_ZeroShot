import os, sys
import os.path as osp
from bt_datasets.imdb import imdb
from bt_net.config import cfg
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import cPickle
from random import shuffle
import subprocess
import xml.etree.ElementTree as ET

import linecache

class nus_wide(imdb):
    def __init__(self, image_set):
        imdb.__init__(self, 'nus_wide_' + image_set)
        self._image_set = image_set

        self._devkit_path = osp.join(cfg.IMAGE_DATA_DIR, 'nus_wide')
        self.image_list = osp.join(self._devkit_path, 'ImageList')
        self._data_path = osp.join(self._devkit_path, 'Images')

        if 'Train' in self._image_set:
            label_file_1k = osp.join(self._devkit_path, 'Concepts', 'Train_Tags920_clean')
            self._classes = self._generate_class_list(label_file_1k)
        elif "_us_" in self._image_set:
            label_file_1k = osp.join(self._devkit_path, 'Concepts', 'Train_Tags920_clean')
            label_file_81 = osp.join(self._devkit_path, 'Concepts', 'Concepts81.txt')
            self._classes = self._generate_class_list(label_file_1k, label_file_81)
        elif "_u_" in self._image_set:
            label_file_81 = osp.join(self._devkit_path, 'Concepts', 'Concepts81.txt')
            print "Generating class list from label_file concepts, only unseen"
            self._classes = self._generate_class_list(label_file_81)
        elif '1k' in self._image_set:
            label_file_1k = osp.join(self._devkit_path, 'Concepts', 'TagList1k.txt')
            self._classes = self._generate_class_list(label_file_1k)
            print "classes", len(self._classes)

        if "train_clean" in self._image_set:
            self.label_file = osp.join(self._devkit_path,"Train_Tags81_clean")
        else:
            self.label_file = osp.join(self._devkit_path,"Test_Tags81_clean.txt")

        print("Num classes: {}").format(len(self._classes))
        
        self._image_index = self._load_image_set_index()
        self._image_ext = ['.JPG']



        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert osp.exists(self._devkit_path), \
                'Devkit path does not exist: {}'.format(self._devkit_path)
        assert osp.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_index_path(self, i):
        return self._image_index[i][0]
        
    def _generate_class_list(self, filename, file_name=None):
        self._classes = ()
        print filename
        with open(filename) as f:
            for line in f:
                self._classes = self._classes + (line.strip(),)
        if file_name is not None:
            with open(file_name) as file:
                for line in file:
                    lbl = line.strip()
                    if lbl not in self._classes:
                        self._classes = self._classes + (lbl, )
        return self._classes

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = osp.join(self._data_path, index[0])
        assert osp.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def load_classes(self):
        return self.classes

    def gt_at(self,i):
        data = self._image_index[i]
        image_data = data[0]
        lbls = [x.strip() for x in data[1:]]
        return lbls

    def get_image_data_clean(self, index, lang_db):
        data = self._image_index[index]
        image_data = data[0]
        lbls = [x.strip() for x in data[1:]]
        img_path = osp.join(self._data_path, image_data)
        word_vectors = []
        for lbl in lbls:
            word_vec = lang_db.word_vector(lbl)
            word_vectors.append(word_vec)
        return img_path , word_vectors, lbls
    '''
    def load_val_data(self, lang_db, size= 10000):
        data = shuffle(self._image_index)

        shuffle(images_data)
        while 1:
            for img_data in images_data:
                img_path = osp.join(val_folder_path, img_data[0] + self._image_ext[0])
                x = read_img(img_path)
                word_vec = lang_db.word_vector(img_data[1])
                yield np.expand_dims(x, axis=0), np.expand_dims(word_vec, axis=0)
    '''
    def get_labels(self, space):

        if space == 2:  # If space is 2, include seen labels in space
            label_file_1k = osp.join(self._devkit_path, "Concepts", "Train_Tags920_clean")
            label_file_81 = osp.join(self._devkit_path, "Concepts", "Concepts81.txt")
            labels = self._generate_class_list(label_file_1k, label_file_81)
            print("the labels", len(labels))
        if space == 1:
            label_file_81 = osp.join(self._devkit_path, "Concepts", "Concepts81.txt")
            labels = self._generate_class_list(label_file_81)[1:]
            print("the labels", len(labels))
        print(labels)
        return labels

    def load_val_data(self, lang_db):
        val_file_set = osp.join(self._devkit_path, 'ImageSets/CLS-LOC/val_bts.txt')
        val_folder_path = osp.join(self._devkit_path, 'Data', 'CLS-LOC', 'val')
        with open(val_file_set) as f:
            images_data = [line.split() for line in f]
            shuffle(images_data)
            while 1:
                for img_data in images_data:
                    img_path = osp.join(val_folder_path, img_data[0] + self._image_ext[0])
                    x = read_img(img_path)
                    word_vec = lang_db.word_vector(img_data[1])
                    yield np.expand_dims(x, axis=0), np.expand_dims(word_vec, axis=0)
    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageList/TrainImageList.txt
        image_set_file = osp.join(self._devkit_path, 'ImageList', self._image_set + '.txt')
        assert osp.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [line.split(",") for line in f]
        return image_index

    def _load_gt_classes(self, index):
        """
        Load image and groundtruth classes  from txt files of nus-wide.
        """
        index = index[0].split('/')
        gt_classes = index[0]
        return {'gt_classes': gt_classes}
