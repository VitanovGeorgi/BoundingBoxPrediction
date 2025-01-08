"""
    A main entry point for the models classes.
"""

import torch

from omegaconf import DictConfig

from models.fastRCNN import get_FastRCNN
from models.VGG16 import get_VGG16
from models.customCNN import get_customCNN




def create_model(cfg: DictConfig) -> torch.nn.Module:
    """
    Create the model based on the configuration file.

    Args:
        cfg (DictConfig): The configuration dictionary.

    Returns:
        torch.nn.Module: The model.
    """
    if cfg.model.model == 'resnet18':
        model = get_FastRCNN(cfg.model.num_classes)
    elif cfg.model.model == 'fasterrcnn_resnet50_fpn':
        raise NotImplementedError("Model not implemented yet!")
    elif cfg.model.model == 'vgg16':
        model = get_VGG16()
    elif cfg.model.model == 'custom':
        model = get_customCNN()
    else:
        raise ValueError(f"Model {cfg.model.name} not recognized.")

    return model

