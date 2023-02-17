from mmcv.runner import load_checkpoint as mmcv_load_checkpoint
from mmcv.runner.checkpoint import load_url_dist
import urllib
import os


dirname = os.path.dirname(__file__)
model_path_NTU = os.path.join(dirname, 'model_NTU.pth')
model_path_Kinect= os.path.join(dirname, 'model_Kinect.pth')


mmskeleton_model_urls = {
    'ddgcn/ntu-xview': model_path_NTU,
    'ddgcn/kinetics-skeleton': model_path_Kinect,
    }




def load_checkpoint(model, filename, *args, **kwargs):
    if filename.startswith('DDGCN://'):
        
        model_name = filename[8:]
        model_url = (mmskeleton_model_urls[model_name])
        checkpoint = mmcv_load_checkpoint(model, model_url, *args, **kwargs)
        return checkpoint

def get_mmskeleton_url(filename):
    if filename.startswith('DDGCN://'):
        model_name = filename[13:]
        model_url = (mmskeleton_model_urls[model_name])
        return model_url
    return filename


def cache_checkpoint(filename):
    try:
        filename = get_mmskeleton_url(filename)
        load_url_dist(get_mmskeleton_url(filename))
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        raise Exception(url_error_message.format(filename)) from e


url_error_message = """

"""
