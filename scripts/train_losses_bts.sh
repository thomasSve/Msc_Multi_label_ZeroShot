#!/bin/sh
# Usage:
# ./scripts/train_losses.sh DATASET LANGUAGE_MODEL
# DATASET is either imagenet or imagenet1k.
#
# Example:
# ./scripts/train_losses_bts.sh imagenet1k glove_wiki_300

set -x
set -e

export PYTHONUNBUFFERED='True'

DATASET=$1
LM=$2

case $DATASET in
    imagenet)
	TRAIN_IMDB="imagenet_train_bts"
	TEST_IMDB="imagenet_test_zsl_bts"
	;;
    imagenet1k)
	TRAIN_IMDB="imagenet1k_train_bts"
	TEST_IMDB="imagenet1k_test_zsl_bts"
	;;
    *)
	echo "No dataset given"
	exit
	;;
esac

for loss in 'hinge' 'cosine_proximity' 'squared_hinge'
do
    # Train model
    time python tools/train_brute_force.py \
	 --imdb ${TRAIN_IMDB} \
	 --lm ${LM} \
	 --loss ${loss} \
	 --iters 10000

    # Make prediction
    time python tools/test_brute_force.py \
	 --ckpt output/bts_ckpt/${TRAIN_IMDB}/model_${TRAIN_IMDB}_${LM}_${loss}_l2_.hdf5 \
	 --imdb ${TEST_IMDB} \
	 --lm ${LM} \
	 --singlelabel_predict \
	 --space 1

done


# Also special case euclidean distance
loss='euclidean'
time python tools/test_brute_force.py \
     --ckpt output/bts_ckpt/${TRAIN_IMDB}/model_${TRAIN_IMDB}_${LM}_${loss}_l2_adam.hdf5 \
     --imdb ${TEST_IMDB} \
     --lm ${LM} \
     --singlelabel_predict

time python tools/test_brute_force.py \
	 --ckpt output/bts_ckpt/${TRAIN_IMDB}/model_${TRAIN_IMDB}_${LM}_${loss}_l2_.hdf5 \
	 --imdb ${TEST_IMDB} \
	 --lm ${LM} \
	 --euc_loss \
	 --singlelabel_predict \
	 --space 1
