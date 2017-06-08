"""
Test the brute-force zero shot. 
1. The testing part splits an image into many boxes, and runs each box through the trained model. 
2. Each box gets labeled by the model, using euclidean distance to find closest label in the space. 
3. After each box is labeled, returns the top-5 predicted labels (if there is less than 5 predicted in total, returned the labels predicted)
4. This way the model predicts in a zero-shot manner, as it will predict labels that exist in the word semantic space, but was not in the trained dataset.

Example run: 
python tools/test_brute_force.py --lm glove_wiki_300 --imdb imagenet_zs --ckpt output/train_bts/model_glove_wiki_300.hdf5 --boxes faster_rcnn

python tools/test_brute_force.py --ckpt output/bts_ckpt/${TRAIN_IMDB}/model_${TRAIN_IMDB}_${LM}_${loss}.hdf5 --imdb ${TEST_IMDB} --lm glove_wiki_300 --singlelabel_predict 
"""
import _init_paths
import argparse
import sys
import numpy as np
from bt_net.test_bts import test_bts
from language_models.language_factory import get_language_model
from bt_datasets.factory import get_imdb

def parse_args():
    parser = argparse.ArgumentParser(description='Train the Brute-Force BTnet model')
    parser.add_argument('--lm', dest='lang_name',
                        help='language model to use (default: glove_wiki_300)',
                        default='glove_wiki_300', type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='imagenet_train', type=str)
    parser.add_argument('--ckpt', dest='ckpt',
                        help='trained model to perform prediction on', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--euc_loss', dest='euc_loss',
                        help='if euclidean_loss is used', action='store_true')
    parser.add_argument('--singlelabel_predict', dest='singlelabel_predict',
                        help='if you want to predict singlelabel', action='store_true')
    parser.add_argument('--space', dest='space',
                        help='select word vector space to predict on: 0: all of wikipedia, 1: only unseen labels, 2:  seen + unseen (default: unseen+seen).', default=2, type=int)
    parser.add_argument('--boxes', dest='boxes',
                        help='predict using generated boxes faster_rcnn or yolo (default: random)', default='random', type=str)



    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if not args.randomize:
        # fix the random seeds (numpy) for reproducibility
        np.random.seed(42)
    
    lang_db = get_language_model(args.lang_name)
    imdb = get_imdb(args.imdb_name)
    
    assert 0 <= args.space <= 2 , \
        'Space has to be either 0, 1 or 2'

    if args.space != 0:
        words = imdb.get_labels(args.space)
    else:
        space = 'all'
        words = []

    lang_db.build_tree(args.space, words, args.imdb_name)
    test_bts(lang_db, imdb, args.ckpt, args.euc_loss, args.singlelabel_predict, args.space, args.boxes)
