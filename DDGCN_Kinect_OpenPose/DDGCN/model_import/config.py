import os
from mmcv import Config as BaseConfig
from DDGCN.version import mmskl_home

#for loading the config file
class Config(BaseConfig):
    @staticmethod
    def fromfile(filename):
        try:
            return BaseConfig.fromfile(filename)
        except:
            return BaseConfig.fromfile(os.path.join(mmskl_home, filename))
