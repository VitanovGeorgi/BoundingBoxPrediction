import os
import pickle
from PIL import Image
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from omegaconf import DictConfig

from datasets.probe_images import get_probe_images_datasets


def collate_resnet18(batch):
    """
        Collate function for the resnet18 model, because we additionally need the targets to be of the following form:
        https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html

        We need to have these functions defined outside of the get_collate_function in order to have them be pickable.
    """
    imgs = []
    imgs_names = []
    targets = []
    for img, img_name, target in batch:
        imgs.append(img)
        imgs_names.append(img_name)
        targets.append(
            {
                "boxes": target["bbox"].unsqueeze(0), # So far the tensor was [4], we need it [N, 4], but N=1 for us (num_classes/labels - 1)
                "labels": torch.tensor([1], dtype=torch.int64)
            }
        )
    return imgs, imgs_names, targets

def collate_fn(batch):
    return batch

def get_collate_function(cfg: DictConfig):
    """
        Get the collate function based on the configuration file. We need this because the original data's bounding boxes are of type x, y, width, height.
    For some models (resnet50), we need them of type x1, y1, x2, y2. But we also need them to have labels, which we don't have in the original data.
    In order to disentangle the two, we deal with the creation of the x1, y1, x2, y2 bounding boxes in the probe_images.py file. And here we deal with the
    creation of the labels. This way we could use the functions from probe_images.py in other projects, where we don't have the requirements for the
    resnet50 model.
    """

    if cfg.model.model == 'resnet18':
        return collate_resnet18
        
    return collate_fn



def get_data_loaders(cfg: DictConfig, gen: torch.Generator) -> tuple:
    """
    Parse the config file, and based on it, return the relevant datasets. From it, create the appropriate data loaders.
    Prepare the data loaders for the training, validation, and test sets.

    Args:
        cfg (DictConfig): The configuration dictionary.
        gen (torch.Generator): The random generator.

    Returns:
        tuple: The training, validation, and test data loaders
    """

    # Get the appropriate dataset
    if cfg.data.name == 'probe_images':
        print("Loading probe images dataset.")
        train_set, validation_set, test_set = get_probe_images_datasets(cfg)
    else:
        raise ValueError(f"Dataset {cfg.dataset.name} not recognized.")

    # --------------------------
    # Create the data loaders
    # --------------------------

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        generator=gen,
        collate_fn=get_collate_function(cfg)
    )
    val_loader = DataLoader(
        validation_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        generator=gen,
        collate_fn=get_collate_function(cfg)
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        generator=gen,
        collate_fn=get_collate_function(cfg)
    )

    return train_loader, val_loader, test_loader





