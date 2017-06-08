import os
from bt_datasets.imdb import imdb
import numpy as np
import scipy.sparse
import scipy.io as sio
import os.path as osp
from bt_net.config import cfg



class open_images(imdb):
    def __init__(self, image_set):
        imdb.__init__(self, image_set)
        self._image_set = image_set
        self._devkit_path = os.path.join(cfg.IMAGE_DATA_DIR, 'openimages')
        self._data_path = os.path.join(self._devkit_path)

        if 'train' in self._image_set:
            train = osp.join(self._devkit_path,'oi_train_classes.txt')
            self._classes = self._generate_class_list(train)
        elif 'test' in self._image_set:
            test = osp.join(self._devkit_path, 'oi_test_classes.txt')
            self._classes = self._generate_class_list(test)
        elif 'validation' in self._image_set:
            test = osp.join(self._devkit_path, 'oi_test_classes.txt')
            self._classes = self._generate_class_list(test)
        self._image_index = self._load_image_set_index()


    
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def _generate_class_list(self,filename, file_name=None):
        self._classes = ('__background__',)
        with open(filename) as f:
            for line in f:
                self._classes = self._classes + (line.strip(),)
        return self._classes

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def get_image_data_clean_train(self, index, lang_db):
        '''

        :param index: Take index 0 - imdb.num_images
        :param lang_db: Can be glove_wiki_300D for example
        :return: path for image, word_vector and name of word_vectors
        '''
        data = self._image_index[index]
        image_data = data[0]
        lbls = data[1:]

        img_path = osp.join(self._data_path, image_data)
        word_vectors = []
        for lbl in lbls:
            word_vec = lang_db.word_vector(lbl)
            word_vectors.append(word_vec)
        return img_path, word_vectors, lbls

    def get_image_data_clean(self, index, lang_db):
        '''
        :param index: Take index 0 - imdb.num_images
        :param lang_db: Can be glove_wiki_300D for example
        :return: path for image and word_vector
        '''
        data = self._image_index[index]
        image_data = data[0]
        lbls = data[1:]
        img_path = osp.join(self._data_path, image_data)
        word_vectors = []
        for lbl in lbls:
            word_vec = lang_db.word_vector(lbl)
            word_vectors.append(word_vec)
        return img_path, word_vectors

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path,
                                  index)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path
        
    def _load_image_set_index(self):
        """ 
        Load the indexes listed in this dataset's image set file
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt
        image_set_file = os.path.join(self._data_path, self._image_set + '.txt')
        assert os.path.exists(image_set_file), 'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip().split(",") for x in f]
        return image_index


