#!/usr/bin/env python
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.losses import cosine_proximity, hinge
from keras import backend as K
from preprocessing.preprocess_images import read_img
from bt_net.generate_boxes import generate_n_boxes
from bt_net.faster_rcnn_boxes import load_frcnn_boxes, boxes_to_images, load_all_boxes_yolo,load_image_boxes_yolo

from multiprocessing import Pool

import os
import os.path as osp
import numpy as np
from random import shuffle
import cv2


def load_data(imdb, lang_db, batch_size = 32):
    total_batch = int(imdb.num_images / batch_size)
    index = [ix for ix in range(imdb.num_images)]

    while 1:
        shuffle(index)
        i = 0
        for _ in range(total_batch):
            X_train = []
            y_train = []
            for j in range(batch_size):
                x_path, y = imdb.get_image_data_clean(index[i], lang_db)
                x = read_img(x_path)
                X_train.append(x)
                y_train.append(y)
                i = i + 1
            yield np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32)

def create_best(boxes, lbl_features, indices):
    new_box =[]
    new_lbls = []
    for i,index in enumerate(indices):
        new_box.append(boxes[index])
        new_lbls.append(lbl_features[i])
    return new_box, new_lbls

def load_multilabel_data(imdb, lang_db, pretrained, box_method, batch_size = 8):
    total_batch = int(imdb.num_images / batch_size)
    index = [ix for ix in range(imdb.num_images)]

    if box_method == 'random':
        pool = Pool(4)
    elif box_method == 'frcnn':
        print "Loading Faster R-CNN boxes..."
        all_boxes = load_frcnn_boxes(imdb)
    elif box_method == 'yolo':
        print "Loading YOLO boxes..."
        all_boxes = load_all_boxes_yolo(imdb)

    while 1:
        shuffle(index)
        i = 0
        for _ in range(total_batch):
            X_train = []
            y_train = []
            j = 0
            while j < batch_size:
                if i >= len(index): i = 0
                x_path, y_s, g_t = imdb.get_image_data_clean(index[i], lang_db)
                img = cv2.imread(x_path)
                if box_method == 'random':
                    boxes = generate_n_boxes(img, pool=pool)
                    k = 1
                elif box_method == 'frcnn':
                    boxes = boxes_to_images(img, all_boxes[index[i]])
                    k = 1
                elif box_method == "yolo":
                    boxes = load_image_boxes_yolo(img, all_boxes, x_path)
                    k = 1
                    if len(boxes) == 0:
                        i = i + 1
                        '''
                        If the path does not exist, len is 0.
                        Jump over this picture.

                        To jump over the picture next time, increment i.

                        '''
                        continue
                j += 1 # Only increment j, when actually appending
                pred_vects = pretrained.predict_on_batch(boxes)
                best_indices, lbl_features = lang_db.get_best_match(pred_vects, y_s, g_t,k)
                best_box, best_lbl_feat = create_best(boxes, lbl_features, best_indices)
                X_train += best_box
                y_train += best_lbl_feat
                i = i + 1
            yield np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32)

def euclidean_distance(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=1, keepdims=True))

def define_network(vector_size, loss):
    base_model = InceptionV3(weights='imagenet', include_top=True)

    for layer in base_model.layers: # Freeze layers in pretrained model
        layer.trainable = False
    
    # fully-connected layer to predict 
    x = Dense(4096, activation='relu', name='fc1')(base_model.layers[-2].output)
    x = Dense(8096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(2048,activation='relu', name='fc3')(x)
    predictions = Dense(vector_size, activation='relu')(x)
    l2 = Lambda(lambda x: K.l2_normalize(x, axis=1))(predictions)
    model = Model(inputs=base_model.inputs, outputs=l2)

    optimizer = 'adam'
    if loss == 'euclidean':
        model.compile(optimizer = optimizer, loss = euclidean_distance)
    else:
        model.compile(optimizer = optimizer, loss = loss)
        
    return model

def train_multilabel_bts(lang_db, imdb, pretrained, max_iters = 1000, loss_func = 'squared_hinge', box_method = 'random'):
    # Create callback_list.
    dir_path = osp.join('output', 'bts_ckpt', imdb.name)
    tensor_path = osp.join(dir_path, 'log_dir')
    if not osp.exists(dir_path):
        os.makedirs(dir_path)
    if not osp.exists(tensor_path):
        os.makedirs(tensor_path)

    ckpt_save = osp.join(dir_path, lang_db.name + '_multi_label_fixed_' + 'weights-{epoch:02d}.hdf5')
    checkpoint = ModelCheckpoint(ckpt_save, monitor='loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=0, mode='auto')
    tensorboard = TensorBoard(log_dir=dir_path, histogram_freq=2000, write_graph=True, write_images=False)
    callback_list = [checkpoint, early_stop, tensorboard]
    pretrained.fit_generator(load_multilabel_data(imdb, lang_db, pretrained, box_method),
                             steps_per_epoch = 5000,
                             epochs = max_iters,
                             verbose = 1,
                             callbacks = callback_list,
                             workers = 1)

    pretrained.save(osp.join(dir_path, 'model_fixed' + imdb.name + '_' + lang_db.name + '_ML_' + box_method + '_' + loss_func + '.hdf5'))

def train_bts(lang_db, imdb, max_iters = 1000, loss = 'squared_hinge'):
    # Define network
    model = define_network(lang_db.vector_size, loss)

    #model = load_model(osp.join('output', 'bts_ckpt', 'imagenet1k_train_bts', 'glove_wiki_300_hinge_weights-03.hdf5'))

    # Create callback_list.
    dir_path = osp.join('output', 'bts_ckpt', imdb.name)
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    log_dir = osp.join('output', 'bts_logs', imdb.name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
        
    ckpt_save = osp.join(dir_path, lang_db.name + "_" + loss + "_weights-{epoch:02d}.hdf5")
    checkpoint = ModelCheckpoint(ckpt_save, monitor='val_loss', verbose=1, save_best_only = True)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
    
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)
    callback_list = [checkpoint, early_stop, tensorboard]
    model.fit_generator(load_data(imdb, lang_db),
                        steps_per_epoch = 5000,
                        epochs = max_iters,
                        verbose = 1,
                        validation_data = imdb.load_val_data(lang_db),
                        validation_steps = 20000, # number of images to validate on
                        callbacks = callback_list,
                        workers = 1)

    model.save(osp.join(dir_path, 'model_'  + imdb.name + '_' + lang_db.name + '_' + loss + '_l2.hdf5'))
