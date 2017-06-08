import _init_paths
import sys
import argparse
from bt_net.visualize_graph import vis_graph

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate results')
    parser.add_argument('--lm', dest='lang_name',
                        help='language model to use',
                        default='glove_wiki', type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to use',
                        default='imagenet_train', type=str)
    parser.add_argument('--vis', dest='vis',
                        help='select graph to display',
                        default='vis_distance', type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ =='__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    vis_graph(args.lang_name, args.imdb_name, args.vis)
