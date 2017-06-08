"""

Train brute force Zero Shot Classification

Example: python tools/train_brute_force.py --imdb imagenet1k_train_bts --lm glove_wiki_300D --loss squared_hinge --iters 10000

"""
import _init_paths
import argparse
import sys
import numpy as np
from bt_net.train_bts import train_bts
from language_models.language_factory import get_language_model
from bt_datasets.factory import get_imdb

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
    parser.add_argument('--loss', dest='loss',
                        help='loss function to run',
                        default='hinge', type=str)

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
    
    train_bts(lang_db, imdb, max_iters = args.max_iters, loss = args.loss)
    
