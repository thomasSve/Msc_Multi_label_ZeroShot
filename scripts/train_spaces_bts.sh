#!/bin/sh
# Usage:
# ./scripts/train_spaces_bts.sh DATASET LOSS function
# DATASET is either imagenet or imagenet1k.
# 
# Example:
# ./scripts/train_spaces_bts.sh imagenet1k squared_hinge

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

for LM in 'w2v_wiki_150D' 'w2v_wiki_300D'
do
    # Train model
    time python tools/train_brute_force.py \
	 --imdb ${TRAIN_IMDB} \
    	 --lm ${LM} \
	 --loss ${LOSS} \
	 --iters 10000
done
