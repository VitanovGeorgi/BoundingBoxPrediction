"""
    A main entry point for the models classes.
"""

import os
import torch

from omegaconf import DictConfig

from models.fastRNN import get_FastRCNN




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
    else:
        raise ValueError(f"Model {cfg.model.name} not recognized.")

    return model

