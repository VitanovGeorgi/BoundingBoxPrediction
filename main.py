
import argparse
from pathlib import Path
import time
import uuid

import torch

from utils.utils import load_config, load_config_omega
from utils.workers import train







parser = argparse.ArgumentParser(description='Robotics Engineer Intern - AI Defect Detection - Flyability')
parser.add_argument('--config', default='configs/default.yaml', type=str, help='config file path')



if __name__ == '__main__':
    args = parser.parse_args()
    cfg = load_config_omega(args.config)

    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)
    print("Config:", cfg)

    train(cfg)
