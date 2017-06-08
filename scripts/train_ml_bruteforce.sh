#!/bin/sh

set -x
set -e
v='w2v_wiki_300D glove_pretrained glove_pretrained'
if [ 1 -eq 0 ]; then


for img_mod in 'nus_wide_Train_zs_920_img_lbl'
do
    for lang_mod in 'w2v_pretrained' 'fast_eng' 'w2v_wiki_300D' 'glove_wiki_300D'
    do
         python tools/train_ml_brute_force.py \
         --lm ${lang_mod}\
         --imdb ${img_mod}\
         --iters 100000\
         --loss sq_hinge
    done
done


for img_mod in 'nus_wide_train_1k'
do
    for lang_mod in  'fast_eng' 'w2v_wiki_300D' 'glove_wiki_300D'
    do
         python tools/train_ml_brute_force.py \
         --lm ${lang_mod}\
         --imdb ${img_mod}\
         --iters 100000
    done
done

fi
if [ 1 -eq 0 ]; then

for img_mod in 'nus_wide_Train_zs_920_img_lbl' 'nus_wide_train_1k'
do
    for lang_mod in 'glove_wiki_300D'
    do
        python tools/train_ml_brute_force.py\
        --lm ${lang_mod}\
        --imdb ${img_mod}\
        --iters 100000\
        --loss sq_hinge_\
        --model model_imagenet1k_train_bts_glove_wiki_300D_squared_hinge_l2.hdf5
    done

done


for img_mod in 'nus_wide_train_1k'
do
    for lang_mod in 'w2v_wiki_300D'
    do
        python tools/train_ml_brute_force.py\
        --lm ${lang_mod}\
        --imdb ${img_mod}\
        --iters 100000\
        --loss sq_hinge_${img_mod}\
        --model /media/bjotta/13f2cffb-0a7d-41b9-946f-36d679d1e9f6/home/fast-rcnn/Zero_Shot_Multi_Label/snapshots/model_imagenet1k_train_bts_w2v_wiki_300D_squared_hinge_l2.hdf5
    done

done


fi
for img_mod in 'nus_wide_Train_zs_920_img_lbl'
do
    for lang_mod in 'w2v_wiki_300D'
    do
        python tools/train_ml_brute_force.py\
        --lm ${lang_mod}\
        --imdb ${img_mod}\
        --iters 100000\
        --loss sq_hinge\
        --boxes yolo \
        --model /media/bjotta/13f2cffb-0a7d-41b9-946f-36d679d1e9f6/home/fast-rcnn/Zero_Shot_Multi_Label/snapshots/model_imagenet1k_train_bts_w2v_wiki_300D_squared_hinge_l2.hdf5
    done

done


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
                        help='method to generate boxes (random, frcnn, yolo)', type=str)





