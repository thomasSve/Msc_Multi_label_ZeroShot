import _init_paths
import argparse
import sys
from bt_net.evaluate_results import *
from language_models.language_factory import get_language_model

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate results')
    parser.add_argument('--results', dest='fn',
                        help='location of result file', type=str)
    parser.add_argument('--k', dest='k',
                        help='top-k value', type=str)
    parser.add_argument('--method', dest='method',
                        help='select method: flat&MAP, pclass or avgeuc',
                        default=None, type=str)
    parser.add_argument('--lm', dest='lang_mod',
                        help='Initiate which language model to use with euc_distance',
                        default='w2v_wiki_300D', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()
    print('Called with args: ')
    print args

    if args.method == 'pclass':
        evaluate_pr_class(args.fn)
    elif args.method == 'apredict':
        show_actual_predicted(args.fn)
    elif args.method == 'avg_class':
        average_gt_classes(args.fn)
    elif args.method == 'avgeuc':
        average_cosine(args.fn, get_language_model(args.lang_mod))
    else:
        evaluate_flat_map(args.fn)
