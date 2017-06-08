import _init_paths
import argparse
import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.manifold import TSNE
from language_models.language_factory import get_language_model
from bt_datasets.factory import get_imdb

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize the predict space')
    parser.add_argument('--lm', dest='lang_name',
                        help='language model to use',
                        default='glove_wiki', type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to predict on',
                        default='imagenet1k_test_zsl', type=str)
    parser.add_argument('--space', dest='space',
                        help='predict space to visualize: 0: all of wikipedia, 1: only unseen labels, 2:  seen + unseen (default: unseen+seen).', default=2, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

 
def main():

    args = parse_args()

    print('Called with args:')
    print(args)
    lang_db = get_language_model(args.lang_name)
    imdb = get_imdb(args.imdb_name)

    # Get words in space
    vocabulary = imdb.get_labels(args.space)

    # Get features for words
    wv = [lang_db.word_vector(w) for w in vocabulary]
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy import spatial
    #spatial.distance.cosine(dataSetI, dataSetII)
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(wv)
 
    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()

if __name__ == '__main__':
    main()
