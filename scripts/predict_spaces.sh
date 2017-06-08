#!/bin/sh
# Usage:
# ./scripts/predict_spaces_bts.sh DATASET LOSS function
# DATASET is either imagenet or imagenet1k.
# 
# Example:
# ./scripts/predict_spaces_bts.sh imagenet1k squared_hinge

set -x
set -e

export PYTHONUNBUFFERED='True'

DATASET=$1
LOSS=$2

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

for LM in 'glove_wiki_300D' 'w2v_wiki_300D'
do
    # Train model
    time python tools/test_brute_force.py \
	 --ckpt output/bts_ckpt/${TRAIN_IMDB}/model_${TRAIN_IMDB}_${LM}_${LOSS}_l2.hdf5 \
	 --imdb ${TEST_IMDB} \
    	 --lm ${LM} \
#	 --singlelabel_predict \
	 --space 1
done
