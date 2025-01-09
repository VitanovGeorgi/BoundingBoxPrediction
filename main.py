
import argparse
from pathlib import Path
import time
import uuid

import torch
import hydra
from omegaconf import DictConfig

from utils.utils import init_distributed_mode, load_config_omega
from utils.workers import train

import pdb





parser = argparse.ArgumentParser(description='Robotics Engineer Intern - AI Defect Detection - Flyability')
parser.add_argument('--config', default='configs/default.yaml', type=str, help='config file path')


# @hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):

    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)
    print("Config:", cfg)

    if cfg.distributed.use_distributed:
        init_distributed_mode(cfg)
    # pdb.set_trace()
    train(cfg)



if __name__ == '__main__':
    args = parser.parse_args()
    cfg = load_config_omega(args.config)
    main(cfg)
    