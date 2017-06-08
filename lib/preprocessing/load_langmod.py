# /usr/local/bin/python
# Made by Bjoernar Remmen
# Short is better
import csv
import time

import h5py
from glove import Glove

root = "../../"
glove_vec_fold = "data/lm_data/glove/"

import numpy as np
import math
from sklearn.preprocessing import normalize
#from keras import backend as K


def load_glove(name):
    model = Glove.load_stanford(root + glove_vec_fold + name)
    return model

def get_features(model):
    '''
    :param glove_model:
    :return list of tuple(word,feature):
    '''
    features = []
    words = []
    for word, index in model.dictionary.iteritems():
        features.append(model.word_vectors[index])
        words.append(word)

    return features, words


def save_h5py(arrays, string_arrs, names, filename="glove.h5"):
    with h5py.File(filename, "w") as hf:
        for i in range(len(arrays)):
            hf.create_dataset(names[i], data=arrays[i])
            string_dt = h5py.special_dtype(vlen=str)
            hf.create_dataset(names[i] + "_words", data=string_arrs[i], dtype=string_dt)

    return True


def load_h5py(name, filename="glove.h5"):
    with h5py.File(filename, "r") as hf:
        data = hf[name][:]
        string_arr = hf[name + "_words"][:]
    return data, string_arr


def group_list(l, group_size):
    """
    :param l:           list
    :param group_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    for i in xrange(0, len(l), group_size):
        yield l[i:i+group_size]

def load_pretrained():

    #glove_vec = ["glove_wiki_50","glove_wiki_150","glove_wiki_300"]
    glove_vec = ["glove_wiki_300"]
    #glove_vec = ["glove_wiki_50"]
    filename = 'glove_pretrained.h5'
    #import tensorflow as tf
    #sess = tf.InteractiveSession()

    features, words = load_h5py('glove_wiki_300',filename=root + glove_vec_fold + filename)
    filename = 'glove.h5'
    features = normalize(np.array(features), axis=1, norm='l2')
    with h5py.File(root + glove_vec_fold + filename, "w") as hf:
        hf.create_dataset(glove_vec[0], data=features)
        string_dt = h5py.special_dtype(vlen=str)
        hf.create_dataset(glove_vec[0] + "_words", data=words, dtype=string_dt)

    for vec in glove_vec:
        data, words = load_h5py(vec, filename=root + glove_vec_fold + "glove.h5")
        print(data.shape, words.shape)
        time.sleep(5)

if __name__ == "__main__":
    #glove_vec = ["glove_wiki_50","glove_wiki_150","glove_wiki_300"]


    glove_vec = ["glove_wiki_300"]
    filename = 'glove.h5'
    #########

    load_pretrained()


    with h5py.File(root + glove_vec_fold + filename, "w") as hf:
        for vec in glove_vec:
            model = load_glove(vec+".txt")
            #ii = model.most_similar('woman',number=5)
            feature_list, words = get_features(model)
            #feature_list = feature_list.tolist()
            counter = 0
            #for i in range(len(words) - 1, -1, -1):
                #try:
                #    words[i] = words[i].decode('utf-8')
                #except:
                #    print "exception", words[i]
                #    feature_list.pop(i)
                #    words.pop(i)


            feature_list = np.array(feature_list)
            #feature_list = normalize(np.array(feature_list), axis=1, norm='l2')
            #feature_list = K.l2_normalize(np.array(feature_list),axis=1).eval()
            print("adding to word array")
            words = np.array(words, dtype=object)
            del model
            print("writing to file")
            hf.create_dataset(vec, data=feature_list)
            string_dt = h5py.special_dtype(vlen=str)
            hf.create_dataset(vec + "_words", data=words, dtype=string_dt)

        #del model  # Trigger garbage collector to free memory.
        #save_h5py(feature_list, words, vec, filename=root + glove_vec_fold + "glove")



    for vec in glove_vec:
        data, words = load_h5py(vec, filename=root + glove_vec_fold + "glove.h5")
        print(data.shape, words.shape)
        time.sleep(5)

    

    '''

    def get_id_dict(filename=None):
        id_to_word = {}
        with open(filename, 'rb') as f:
            reader = csv.reader(f)

            for line in reader:
                id_to_word[line[0]] = line[1].lower()

        return id_to_word
    def get_oi_labels(id_to_word):
        unique = {}
        open_images = "/media/bjotta/13f2cffb-0a7d-41b9-946f-36d679d1e9f6/home/GloVe/data/machine_ann_2016_08/train/labels.csv"
        with open(open_images, 'rb') as file:
            reader = csv.reader(file)
            reader.next()
            for line in reader:
                label_ID = line[2].lower()
                if id_to_word[label_ID] not in unique:
                    unique[id_to_word[label_ID]] = 1
        return unique.keys()

    id_to_word = get_id_dict(filename="/media/bjotta/13f2cffb-0a7d-41b9-946f-36d679d1e9f6/home/GloVe/data/dict.csv")
    unique_words = get_oi_labels(id_to_word)
    glove_vec = ['glove_oi_50']
    counter = 0
    not_in ={}
    for vec in glove_vec:
        model = load_glove(vec+".txt")
        feature_list, glove_words = get_features(model)
        for word in unique_words:
            if word not in glove_words and word not in not_in:
                not_in[word] =1
                counter +=1
                print word, counter

    '''




