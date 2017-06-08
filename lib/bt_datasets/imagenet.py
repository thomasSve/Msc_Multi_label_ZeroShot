import os, sys
import os.path as osp
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


class imagenet(imdb):
    def __init__(self, image_set):
        imdb.__init__(self, 'imagenet_' + image_set)
        self._image_set = image_set
        self._devkit_path = osp.join(cfg.IMAGE_DATA_DIR, 'imagenet')
        self._data_path = osp.join(self._devkit_path, 'Data/DET', self._image_folder())
        synsets = sio.loadmat(osp.join(self._devkit_path, 'meta_det.mat'))
        self._classes = ('__background__',)
        self._wnid = (0,)
        for i in xrange(200):
                self._classes = self._classes + (synsets['synsets'][0][i][2][0],)
                self._wnid = self._wnid + (synsets['synsets'][0][i][1][0],)
        self._wnid_to_ind = dict(zip(self._wnid, xrange(self.num_classes)))
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = ['.JPEG']
        self._image_index = self._load_image_set_index()
        #print "image_index", self._image_index


        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._devkit_path), \
                'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_index_path(self, i):
        return self._image_index[i][0]

        
    def _image_folder(self):
        image_folder = ""
        if 'train' in self._image_set: return 'train'
        elif 'test_zs' in self._image_set: return 'test_zs'
        elif 'test' in self._image_set: return 'test'
        return 'val'
    
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])
    
    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path,
                                  index[0] + self._image_ext[0])
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt or self._data_path + /ImageSets/zero_shot.txt
        image_set_file = os.path.join(self._devkit_path, 'ImageSets/DET/', self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            image_index = [line.split() for line in f]
        return image_index

    def load_classes(self):
        return self._classes[1:] # Return classes except background

    def load_image_lbl(self):
        i = 0
        index = [ix for ix in range(self.num_images)]
        shuffle(index)
        while i < self.num_images:

            image_dir = self._image_index[index[i]]
            image_path = self.image_path_at(index[i])
            filename = os.path.join(self._devkit_path, 'Annotations', self._image_folder(), image_dir[0] + '.xml')
            tree = ET.parse(filename)

            def get_data_from_tag(node, tag):
                return node.getElementsByTagName(tag)[0].childNodes[0].data

            with open(filename) as f:
                data = minidom.parseString(f.read())

            objs = data.getElementsByTagName('object')
            unique = {}
            for ix, obj in enumerate(objs):
                cls = self._classes[self._wnid_to_ind[
                    str(get_data_from_tag(obj, "name")).lower().strip()]]
                if cls not in unique:
                    unique[cls] = cls

            i += 1
            if i >=(self.num_images):
                i = 0
                shuffle(index)
            yield image_path, [str(x) for x in unique.keys()]

    def get_image_data_clean(self, index, lang_db):
        image_data = self._image_index[index]
        img_path = osp.join(self._data_path, image_data[0] + self._image_ext[0])
        word_vec = lang_db.word_vector(image_data[1])
        return img_path, word_vec
    
    def get_image_label(self, index, lang_db):
        image_dir = self._image_index[index]
        filename = os.path.join(self._devkit_path, 'Annotations/DET/', self._image_folder(), image_dir[0] + '.xml')
        tree = ET.parse(filename)
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        num_objs = len(objs)
        if num_objs != 1: return False

        cls = self._classes[self._wnid_to_ind[str(get_data_from_tag(objs[0], "name")).lower().strip()]]
        y = lang_db.word_vector(cls)
        if y is None: return False
        return [image_dir[0], cls]

        
    def load_val_data(self, lang_db):
        val_file_set = osp.join(self._devkit_path, 'ImageSets/DET/val_bts.txt')
        val_folder_path = osp.join(self._devkit_path, 'Data', 'DET', 'val')
        with open(val_file_set) as f:
            images_data = [line.split() for line in f]
            while 1:
                for img_data in images_data:
                    img_path = osp.join(val_folder_path, img_data[0] + self._image_ext[0])
                    x = read_img(img_path)
                    word_vec = lang_db.word_vector(img_data[1])
                    yield np.expand_dims(x, axis=0), np.expand_dims(word_vec, axis=0)

    def get_image_data_clean(self, index, lang_db):
        image_data = self._image_index[index]
        img_path = osp.join(self._data_path, image_data[0] + self._image_ext[0])
        word_vec = lang_db.word_vector(image_data[1])
        return img_path, word_vec
    
    def get_image_label(self, index, lang_db):
        image_dir = self._image_index[index]
        filename = os.path.join(self._devkit_path, 'Annotations/DET/', self._image_folder(), image_dir[0] + '.xml')
        tree = ET.parse(filename)
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        num_objs = len(objs)
        if num_objs != 1: return False

        cls = self._classes[self._wnid_to_ind[str(get_data_from_tag(objs[0], "name")).lower().strip()]]
        y = lang_db.word_vector(cls)
        if y is None: return False
        return [image_dir[0], cls]

        
    def load_val_data(self, lang_db):
        val_file_set = osp.join(self._devkit_path, 'ImageSets/DET/val_bts.txt')
        val_folder_path = osp.join(self._devkit_path, 'Data', 'DET', 'val')
        with open(val_file_set) as f:
            images_data = [line.split() for line in f]
            shuffle(images_data)
            while 1:
                for img_data in images_data:
                    img_path = osp.join(val_folder_path, img_data[0] + self._image_ext[0])
                    x = read_img(img_path)
                    word_vec = lang_db.word_vector(img_data[1])
                    yield np.expand_dims(x, axis=0), np.expand_dims(word_vec, axis=0)

    def load_generated_box_features(self, proposals_file):
        """ Load the boxes generated by py-faster-rcnn """
        num_images = self.num_images

        # Load pickle with proposals
        proposals_path = os.path.join(self._devkit_path, 'Generated_proposals', proposals_file)
        print(proposals_path)
        with open(proposals_path, 'rb') as input_file:
            proposals = cPickle.load(input_file)

        # proposals has shape (num_images, 2000, 4) and a few images (600, 4)

        num_proposals = len(proposals)
        assert num_images != num_proposals, \
                             'Mismatch between number of images in imagelist and the proposal list, {}, {}'.format(num_images, num_proposals)
        i = 0

        # Clean boxes smaller than 0.3 of image, have top 3 or top 5 of bounding boxes? Or base it on scores?
        for ix in proposals:
            if i%100 == 0:
                ix = np.array(ix)
                print ix[0]
                print("images {}/{}".format(i, num_images))
            i = i + 1 

            
