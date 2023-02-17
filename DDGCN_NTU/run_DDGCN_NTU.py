from DDGCN.model_import import call_obj, parse_cfg
import os.path
from pathlib import Path
import argparse
dirname = os.path.dirname(__file__)
dir_path = Path(__file__).absolute().parent 


def main():


    parser = argparse.ArgumentParser("DDGCN running options")

    parser.add_argument("--pose",
                    type=str,
                    help="Path to the skeleton dir")
    
    args = parser.parse_args()
    if args.rec:
        #print("action recognition")
        cfg_path = (str(dir_path) +   '/configs/recognition/test_NTU.yaml')
    
        cfg = parse_cfg(parser, cfg_path)
        cfg.processor_cfg.dataset_cfg.data_path = args.pose
        call_obj(**cfg.processor_cfg)

               
if __name__ == "__main__":
    main()