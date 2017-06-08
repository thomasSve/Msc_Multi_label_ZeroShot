#!/usr/bin/env python
"""

Train brute force Zero Shot Classification
1. Train a small dense network that learns to train inception features of 1024 down to 300d,
and inputs images in same semantic space as language model

"""
import _init_paths
import argparse
import sys
import numpy as np
from bt_net.train_bts import train_multilabel_bts, euclidean_distance
from language_models.language_factory import get_language_model
from bt_datasets.factory import get_imdb
from keras.models import load_model
import os.path as osp
from bt_net.config import cfg
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session



def parse_args():
    parser = argparse.ArgumentParser(description='Train the Brute-Force BTnet model')
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--lm', dest='lang_name',
                        help='language model to use',
                        default='glove_wiki', type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='imagenet_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--loss', dest='loss_func',
                        help='name of loss in pretrained',
                        default='squared_hinge', type=str)
    parser.add_argument('--model', dest='model',
                        help='pretrained model', type=str)
    parser.add_argument('--boxes', dest='boxes',
                        help='method to generate boxes (random, frcnn, yolo)',
                        default='random', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ =='__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(42)
    
    lang_db = get_language_model(args.lang_name)    
    imdb = get_imdb(args.imdb_name)
    '''
    lang_db.build_tree("nus_wide_train_1k",imdb.load_classes())
    #pretrained = load_model(osp.join(cfg.SNAPSHOT_DIR, "model_train_bts_glove_wiki_300.hdf5"), custom_objects={'euclidean_distance': euclidean_distance})
    #pretrained = load_model(osp.join(cfg.SNAPSHOT_DIR, args.model))
    pretrained = load_model(args.model)

    print("test random",(pretrained.predict_on_batch(np.random.rand(30,299,299,3))).shape)
    train_multilabel_bts(lang_db, imdb, pretrained, max_iters=args.max_iters, loss_func=args.loss_func)
    '''
    pretrained = load_model(args.model)
    print("test random",(pretrained.predict_on_batch(np.random.rand(30,299,299,3))).shape)
    train_multilabel_bts(lang_db, imdb, pretrained, max_iters=args.max_iters, loss_func=args.loss_func, box_method=args.boxes)
