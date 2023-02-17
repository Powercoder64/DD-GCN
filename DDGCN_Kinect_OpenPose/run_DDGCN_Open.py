from DDGCN.model_import import call_obj, parse_cfg
from pathlib import Path
from DDGCN.utils.visualization import visualize
import argparse
from DDGCN.data_import.openpose_convert import OpenPose_convert
from DDGCN.data_import.openpose_convert import OpenPose_convert_vis
import numpy as np

dir_path = Path(__file__).absolute().parent 


def main():

    
    parser = argparse.ArgumentParser("DDGCN running options")
    parser.add_argument("--rec",
                        help="action recognition",
                        action="store_true")
    parser.add_argument("--vis", 
                        help="visualization",
                        action="store_true")
    parser.add_argument("--pose",
                    type=str,
                    help="path to the OpenPose output dir")
    parser.add_argument("--video",
                    type=str,
                    help="path to the video dir") 
    parser.add_argument("--wi",
                    type=str,
                    help="frame width for pose data normalization")
    parser.add_argument("--he",
                    type=str,
                    help="frame height for pose data normalization")                       

    
    args = parser.parse_args()
    
    convert_path = str(dir_path) + '/data/recognition/OpenPose/data_convert/'
    if args.rec:

        cfg_path = (str(dir_path) + '/configs/recognition/test_OpenPose.yaml')
        
        openpose_numpy = OpenPose_convert(args.pose, int(args.wi), int(args.he))
        np.save(convert_path + 'openpose_numpy', openpose_numpy)
    
        cfg = parse_cfg(parser, cfg_path)
        cfg.processor_cfg.dataset_cfg.data_path = convert_path + 'openpose_numpy.npy'
        call_obj(**cfg.processor_cfg)


    elif args.vis:
        joints = OpenPose_convert_vis(args.pose)
        visualize(args.video, joints)
              
if __name__ == "__main__":
    main()