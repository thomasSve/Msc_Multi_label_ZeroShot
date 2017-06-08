### Set up paths

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

## Including lib folder
lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)

zsl_root = osp.join(this_dir, '..')
add_path(zsl_root)
