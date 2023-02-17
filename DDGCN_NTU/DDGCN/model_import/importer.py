import sys
import argparse
from mmcv import Config
#to import diffrent objects defined in the config file
def import_obj(name):
    if not isinstance(name, str):
        raise ImportError('Object name should be a string.')

    if name[0] == '.':
        name = 'DDGCN' + name

    mod_str, _sep, class_str = name.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Object {} cannot be found.'.format(class_str))


def call_obj(name, **kwargs):
    return import_obj(name)(**kwargs)


def set_attr(obj, name, value):
    if not isinstance(name, str):
        raise ImportError('Attribute name should be a string.')

    attr, _sep, others = name.partition('.')
    if others == '':
        setattr(obj, attr, value)
    else:
        set_attr(getattr(obj, attr), others, value)


def get_attr(obj, name):
    if not isinstance(name, str):
        raise ImportError('Attribute name should be a string.')

    attr, _sep, others = name.partition('.')
    if others == '':
        return getattr(obj, attr)
    else:
        return get_attr(getattr(obj, attr), others)
    
 
def parse_cfg(parser, data_path):
    

    cfg = Config.fromfile(data_path)
    
    for key, info in cfg.argparse_cfg.items():
        if 'bind_to' not in info:
            continue
        default = get_attr(cfg, info['bind_to'])
        if 'type' not in info:
            if default is not None:
                info['type'] = type(default)
        kwargs = dict(default=default)
        kwargs.update({k: v for k, v in info.items() if k != 'bind_to'})
        parser.add_argument('--' + key, **kwargs)
    args = parser.parse_args()

    # update config from command line
    for key, info in cfg.argparse_cfg.items():
        if 'bind_to' not in info:
            continue
        value = getattr(args, key)
        set_attr(cfg, info['bind_to'], value)
    
    return cfg

    

