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


class imagenet1k(imdb):
    def __init__(self, image_set):
        imdb.__init__(self, 'imagenet1k_' + image_set)
        self._image_set = image_set
        self._devkit_path = osp.join(cfg.IMAGE_DATA_DIR, 'imagenet1k')
        self._data_path = osp.join(self._devkit_path, 'Data/CLS-LOC', self._image_folder())
        self._synsets = open((osp.join(self._devkit_path, 'map_clsloc.txt'))).readlines()

        self._classes = ('__background__',)
        self._wnid = (0,)
        for i in xrange(1000):
            line = self._synsets[i].rstrip().split(" ")
            self._classes = self._classes + (line[2],)
            self._wnid = self._wnid + (line[0],)
        self._wnid_to_ind = dict(zip(self._wnid, xrange(self.num_classes)))
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = ['.JPEG']
        self._image_index = self._load_image_set_index()
        # print "image_index", self._image_index


        # Specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'top_k': 2000}

        assert os.path.exists(self._devkit_path), \
            'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_index_path(self, i):
        return self._image_index[i][0]

    def _image_folder(self):
        image_folder = ""
        if 'val' in self._image_set:
            return 'val'
        #elif 'test_zs' in self._image_set:
        #    return 'test_zs'
        #elif 'test' in self._image_set:
        #    return 'test'
        return 'train'

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
        image_set_file = osp.join(self._devkit_path, 'ImageSets','CLS-LOC', self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            image_index = [line.split() for line in f]
        return image_index

    def get_labels(self, space):
        labels = []
        imageset_path = osp.join(self._devkit_path,'ImageSets', 'CLS-LOC',)
        unseen_f = open(osp.join(imageset_path, 'test_200_classes.txt'))
        labels = [x.strip() for x in unseen_f]
        
        if space == 2: # If space is 2, include seen labels in space
            seen_f = open(osp.join(imageset_path, 'train_800_classes.txt'))
            for line in seen_f:
                labels.append(line.strip())
            
        return labels
            
    def generate_zs_split(self):
        new_list = deepcopy(self._synsets)
        shuffle(new_list)
        train_split = os.path.join(self._devkit_path,  "train_800_classes.txt")
        test_split = os.path.join(self._devkit_path, "test_200_classes.txt")

        with open(train_split,"w") as train, open(test_split,"w") as test:
            for i in range(800):
                train.write(new_list[i])
            for x in range(800,1000):
                test.write(new_list[x])
        print "Wrote train and test split to:"
        print os.path.join(self._devkit_path, 'ImageSets/CLS-LOC/')

    def clean_zs_split(self, lang_db):
        train_800 = os.path.join(self._devkit_path, "ImageSets", "CLS-LOC", "train_800.txt")
        test_200 = os.path.join(self._devkit_path, "ImageSets", "CLS-LOC", "test_200.txt")
        dict_200 = {}
        dict_800 = {}
        with open(train_800) as train, open(osp.join(self._devkit_path, "train_800_fixed_" + lang_db.name + ".txt"),"w") as train_fix, open("train_800_classes_"+ lang_db.name + ".txt", "w") as train_lbl:
            for line in train:
                wnid = line.rstrip().split("/")[0]
                cls = self._classes[self._wnid_to_ind[wnid]].lower()
                if lang_db.word_vector(cls.lower()) is not None:
                    train_fix.write(line)
                    if cls not in dict_800:
                        dict_800[cls] = 1
                        train_lbl.write(cls + "\n")
        with open(test_200) as train, open(osp.join(self._devkit_path, "test_200_fixed_" + lang_db.name + ".txt"), "w") as test_fix, open("test_200_classes_" + lang_db.name + ".txt", "w") as test_lbl:
            for line in train:
                wnid = line.rstrip().split("/")[0]
                cls = self._classes[self._wnid_to_ind[wnid]].lower()
                if lang_db.word_vector(cls.lower()) is not None:
                    test_fix.write(line)
                    if cls not in dict_200:
                        dict_200[cls] = 1
                        test_lbl.write(cls + "\n")

        print "dict_200", len(dict_200.keys())
        print "dict_800", len(dict_800.keys())




    def zs_text_files(self):
        train_zs = open(os.path.join(self._devkit_path,  "train_800_classes.txt")).readlines()
        train_zs_wnid = [x.split(" ")[0] for x in train_zs]
        test_zs = open(os.path.join(self._devkit_path, "test_200_classes.txt")).readlines()
        test_zs_wnid = [x.split(" ")[0] for x in test_zs]

        #train_split = os.path.join(self._devkit_path, 'ImageSets/CLS-LOC/','train_800.txt')
        #test_split = os.path.join(self._devkit_path, 'ImageSets/CLS-LOC/','test_200.txt')
        train_split = os.path.join(self._devkit_path, 'train_800.txt')
        test_split = os.path.join(self._devkit_path,  'test_200.txt')

        with open(train_split,"w") as train, open(test_split, "w") as test:
            index = [ix for ix in range(self.num_images)]
            for i in range(self.num_images):
                image_dir = self._image_index[index[i]]
                image_dir0 = image_dir[0]
                print image_dir
                wn_id = image_dir0.split("/")[0]
                if wn_id in train_zs_wnid:
                    train.write(" ".join(image_dir) +"\n")
                elif wn_id in test_zs_wnid:
                    test.write(" ".join(image_dir) + "\n")

    def load_classes(self):
        return self._classes[1:]  # Return classes except background

    def load_image_lbl(self):
        i = 0
        index = [ix for ix in range(self.num_images)]
        shuffle(index)
        while i < self.num_images:

            image_dir = self._image_index[index[i]][0]
            wn_id = image_dir.split("/")[0]
            image_path = self.image_path_at(index[i])
            i += 1
            if i >= self.num_images:
                i = 0
                shuffle(index)
            yield image_path, [self._classes[self._wnid_to_ind[wn_id]].lower()]

    def load_image_lbl_imdir(self):
        i = 0
        index = [ix for ix in range(self.num_images)]
        shuffle(index)
        while i < self.num_images:

            image_dir = self._image_index[index[i]][0]
            full_image_dir = self._image_index[index[i]]
            wn_id = image_dir.split("/")[0]
            image_path = self.image_path_at(index[i])
            i += 1
            yield image_path, [self._classes[self._wnid_to_ind[wn_id]].lower()], full_image_dir

    def get_image_data_clean(self, index, lang_db):
        image_data = self._image_index[index]
        img_path = osp.join(self._data_path, image_data[0] + self._image_ext[0])
        word_vec = lang_db.word_vector(image_data[1])
        return img_path, word_vec

    def gt_at(self, index):
        image_dir = self._image_index[index][0]
        wn_id = image_dir.split("/")[0]
        return self._classes[self._wnid_to_ind[wn_id]].lower()
    
    def get_image_label(self, index, lang_db):
        image_dir = self._image_index[index][0]
        wn_id = image_dir.split("/")[0]
        image_path = self.image_path_at(index)

        cls = self._classes[self._wnid_to_ind[wn_id]].lower()
        y = lang_db.word_vector(cls)
        if y is None: return False
        return [image_dir, cls]

    def get_val_label(self, index, lang_db):
        image_dir = self._image_index[index]
        filename = os.path.join(self._devkit_path, 'Annotations/CLS-LOC/', self._image_folder(), image_dir[0] + '.xml')
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
            'Mismatch between number of images in imagelist and the proposal list, {}, {}'.format(num_images,
                                                                                                  num_proposals)
        i = 0

        # Clean boxes smaller than 0.3 of image, have top 3 or top 5 of bounding boxes? Or base it on scores?
        for ix in proposals:
            if i % 100 == 0:
                ix = np.array(ix)
                print ix[0]
                print("images {}/{}".format(i, num_images))
            i = i + 1


