# /usr/local/bin/python3.5
# Made by Bjoernar Remmen
# Short is better
import os.path as osp
from easydict import EasyDict as edict
__C = edict()


cfg = __C

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
__C.DATA_DIR = osp.join(__C.ROOT_DIR, 'data')
__C.LM_DATA_DIR = osp.join(__C.DATA_DIR, 'lm_data')
__C.IMAGE_DATA_DIR = osp.join(__C.DATA_DIR, 'image_data')
__C.SNAPSHOT_DIR = osp.join(__C.ROOT_DIR, "snapshots")
__C.MODEL_DIR = osp.join(__C.ROOT_DIR, "models")
__C.FASTER_RCNN_DIR = osp.join(__C.ROOT_DIR, '..', 'py-faster-rcnn') # Set to py-faster-rcnn root
__C.FASTER_RCNN_LIB = osp.join(__C.FASTER_RCNN_DIR, 'lib')

# Set fixed seed for reproducibility
__C.RNG_SEED = 3



def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
