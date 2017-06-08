import os
import os.path as osp
import numpy as np
from bt_net.config import cfg
from language_models.lmdb import lmdb
from preprocessing.load_langmod import load_h5py
from annoy import AnnoyIndex
from scipy.spatial import ckdtree
import pickle
import operator
from scipy.spatial import distance
from scipy import spatial


class glove_factory(lmdb):
    def __init__(self, corpus, dimension, glove_file='language_models.h5'):
        lmdb.__init__(self, str(corpus))
        self._corpus = corpus
        self._devkit_path = osp.join(cfg.LM_DATA_DIR, 'glove')

        print "feature_name", self.name
        print "devkit_path", self._devkit_path
        self.features, self.words = load_h5py(self.name,
                                             osp.join(self._devkit_path, glove_file))

        assert len(self.features) == len(self.words), \
            'Feature and word array not equal length. feature:{} words: {}'.format(str(len(self.features)), str(len(self.words)))

        assert os.path.exists(self._devkit_path), \
            'Devkit path does not exist: {}'.format(self._devkit_path)

        self._dimension = dimension
        self._vectors = self.vector_dict()

    @property
    def vector_size(self):
        return self._dimension

    def vector_dict(self):
        """
        Return a dictionary with the vector values of the classes
        The i-th class and the i-th feature is the two that correlate
        """
        vector_dict = {}
        for i in range(len(self.words)):
             vector_dict[self.words[i].lower()] = self.features[i]
        return vector_dict

    def build_tree(self, space, words, imdb_name, num_trees = 1000, vector_list = None):
        """
        Build annoy tree to calculate distance, if vector_list = None builds full tree with all words in language model. If not None, builds using the words in the vector list.
        """
        # If pckl exist, load, else build
        tree_path = osp.join(self._devkit_path, self.name + '_' + imdb_name + str(space) + '.ann')
        pckl_path = osp.join(self._devkit_path, self.name + '_' + imdb_name + str(space) + 'array'+".pkl")
        t = AnnoyIndex(self._dimension, metric="euclidean")
        if osp.exists(tree_path):
            print "Tree exist, loading from file..."

            t.load(tree_path)
            self._tree = t
            with open(pckl_path,  'rb') as file:
                self._labels = pickle.load(file)
        else:
            print "Building tree..."

            counter = 0
            word_list = []
            if space == 0:
                for word, feature in self._vectors.iteritems():
                    word_list.append(word)
                    t.add_item(counter,feature)
                    counter += 1
            else:
                for w in words:
                    word_list.append(w)
                    t.add_item(counter, self.word_vector(w))
                    counter += 1

            t.build(num_trees)
            self._tree = t
            self._labels = word_list

            # Save tree
            t.save(tree_path)
            with open(pckl_path, 'wb') as handle:
                pickle.dump(word_list,handle)

    def closest_labels(self, labels_vector, k_closest = 1):
        '''
        :param labels_vector:
        :return: Returns the words of the closest labels
        '''
        nearest = []
        for v in labels_vector:
            nearest.append(self._tree.get_nns_by_vector(v, k_closest, include_distances=False))
        return [[self._labels[x] for x in nearest[y]] for y in range(len(nearest))]

    def get_label_index(self,label):
        for i in range(len(self._labels)):
            if label == self._labels[i]:
                return i

    def get_feature(self,label):
        index = self.get_label_index(label)
        if index is None:
            return index
        else:
            return self._tree.get_item_vector(self.get_label_index(label))

    def is_existing(self,word):
        for i in range(len(self._labels)):
            if word == self._labels[i].lower():
                return i
        return False
    def get_lbl_features(self, pic_labels, k_closest=1):
        label_indices = []

        for x in range(len(pic_labels)):
            temp = self.is_existing(pic_labels[x])
            if temp == False:
                pic_labels.pop(x)
            else:
                label_indices.append(temp)

        nearest = []
        for lbl_ind in label_indices:
            nearest.append(self._tree.get_item_vector(lbl_ind))
        return nearest, pic_labels

    def get_k_smallest(self,distance, indices, k):
        distances = []
        #x = [list(x) for x in zip(*sorted(zip(distance, indices), key=operator.itemgetter(0)))]
        x = [list(x) for x in zip(*sorted(zip(distance, indices)))]
        distances = x[0]
        indices = x[1]
        return indices[-k:]

    def get_closest(self,img_feature, lbl_features):
        smallest_distance = 999999999
        smallest_ind = None
        for i in range(len(lbl_features)):
            value = distance.euclidean(img_feature,lbl_features[i])
            if value < smallest_distance:
                smallest_distance = value
                smallest_ind = i

        return smallest_distance, smallest_ind
    '''
    def get_best_match4(self,pred_vects, lbl_features, labels=None, k=2, num_trees=100):

        #Goal: get k closest for every label


        t = AnnoyIndex(len(lbl_features[0]))
        for i in range(len(pred_vects)):
            t.add_item(i, pred_vects[i])
        t.build(num_trees)
        feature_inds = []
        label_inds = []
        for x in range(len(lbl_features)):
            index = t.get_nns_by_vector(lbl_features[x], k, include_distances=False)
            for ind in index:
                label_inds.append(x)
                feature_inds.append(ind)
        return feature_inds, label_inds
    '''


    def get_best_match(self, pred_vects, lbl_features, labels, k, expand=True, num_trees=100):
        ''''
        Goal: get 2 closest for every label
        if expand = True - Avoid using closest index  in image vectors for another label
        else: Can get same index for another label

        Method to get match. A little faster than original code
        '''
        t = AnnoyIndex(len(lbl_features[0]))
        for i in range(len(pred_vects)):
            t.add_item(i, pred_vects[i])
        t.build(num_trees)
        image_feature_inds = []
        final_lbl_featues = []
        temp = 0
        used_inds = {}
        for x in range(len(lbl_features)):
            if expand == False:
                temp = 2
            else:
                temp += k
            indices = t.get_nns_by_vector(lbl_features[x], temp, include_distances=False)
            add = 0
            for ind in indices:
                if (add > k): break
                if temp == len(pred_vects): # Edge case. If every index is used.
                    final_lbl_featues.append(lbl_features[x])
                    image_feature_inds.append(indices[0])
                    return image_feature_inds, final_lbl_featues
                if ind in used_inds and expand:
                    continue
                else:
                    if ind not in used_inds: used_inds[ind] = 1
                    #label_inds.append(x)
                    final_lbl_featues.append(lbl_features[x])
                    image_feature_inds.append(ind)
                    add += 1
        return image_feature_inds, final_lbl_featues




    def get_best_match_OLD(self, pred_vects, lbl_features, labels, k=2, num_trees=1):
        #print("lblfeature",len(lbl_features),"lbls", len(labels))
        #print("pred_vects", pred_vects)

        # Make one sublist for each label
        all_closest = [[] for x in range(len(lbl_features))]

        # Iterate through all lbl_features
        # For each lbl_feature find distance to every image_vector
        # Add with (distance, image index, lbl index)

        for i in range(len(lbl_features)):
            closest = []
            for ind, element in enumerate(pred_vects):
                value = distance.euclidean(pred_vects[ind], lbl_features[i])
                if len(closest) > 0:
                    added = False
                    for cls_ind in range(len(closest)):
                        if closest[cls_ind][0] > value:
                            # THe value for the distance. Smallest is best
                            # Index for best vector
                            # index for best lbl
                            closest.insert(cls_ind,(value,ind,i))
                            added = True
                            break
                    if added == False:
                        closest.append((value, ind, i))
                else:
                    closest.append((value, ind,i))
                    #pred_vects.pop(ind)
            all_closest[i] = closest

        # Find the lbl that is closest to some ind.
        # This is one of the first in every sub array in the all_closest.
        # This is because the first is always the smallest.
        closest_ind, closest_val = zip(*sorted(enumerate([x[0][0] for x in all_closest]), key=operator.itemgetter(1)))
        closest_ind, closest_val = list(closest_ind),list(closest_val)
        best_indices = []
        temp_best_inds = []
        used_boxes = []
        for ind in closest_ind:
            for clos in all_closest[ind]:
                if len(temp_best_inds) < k and clos[1] not in used_boxes:
                    used_boxes.append(clos[1])
                    temp_best_inds.append(clos)
                else:
                    continue
            best_indices += temp_best_inds
        final_best_ind = []
        final_lbl_features = []
        for best in best_indices:
            final_best_ind.append(best[1])
            final_lbl_features.append(lbl_features[best[2]])
        return final_best_ind, final_lbl_features

    def get_word_features(self):
        return self.features, self.words

    def word_vector(self, label):
        """ Load in the word vector for the given label """
        try:
            return self._vectors[label]
        except:
            print "Missing label: ", label
            return None

