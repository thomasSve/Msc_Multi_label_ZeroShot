#!/usr/bin/env python
"""Factory method for easily getting imdbs by name."""

__sets = {}

from bt_datasets.imagenet import imagenet
from bt_datasets.open_images import open_images
from bt_datasets.nus_wide import nus_wide
from bt_datasets.imagenet1k import imagenet1k
from bt_datasets.places205 import places205
from bt_datasets.places365 import places365



#from bt_datasets.places205 import places205
import numpy as np

# Set up imagenet
for split in ['train', 'train_small', 'train_bts','train_no_extra', 'train_clean_imagelist','train_1','val', 'val_bts','val1', 'val2', 'test', 'test_zs']:
    name = 'imagenet_{}'.format(split)
    __sets[name] = (lambda split=split: imagenet(split))

# Set up Imagenet1k -2012
for split in ['train_cls','test', 'val', 'train_800',
              'test_200', 'train_800_fixed', 'test_200_fixed',
              'train_bts', 'test_zsl_bts', 'val_bts']:
    name = 'imagenet1k_{}'.format(split)
    __sets[name] = (lambda split=split: imagenet1k(split))

# Set up NUS-WIDE
for split in ['Train_zs_920_img_lbl', 'Test_zs_us_img_lbl', 'Test_zs_u_img_lbl', 'train_1k' ]:
    name = 'nus_wide_{}'.format(split)
    __sets[name] = (lambda split=split: nus_wide(split))


# Set up places205
for split in ['train', 'train_8']:
    name = 'places205_{}'.format(split)
    __sets[name] = (lambda split=split: places205(split))

# Set up places365
for split in ['train_zs']:
    name = 'places365_{}'.format(split)
    __sets[name] = (lambda split=split: places365(split))

# Set up for OpenImages
for split in ['train_zs',"test_zs","validation"]:
    name = 'openimages_{}'.format(split)
    __sets[name] = (lambda split=split: open_images(split))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()