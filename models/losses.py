"""
Utility methods for constructing losses.
"""

import torch
from omegaconf import DictConfig



def create_loss(config: DictConfig) -> torch.nn.Module:
    """
    Create the loss function based on the configuration file.

    Args:
        config (DictConfig): The configuration dictionary.

    Returns:
        torch.nn.Module: The loss function.
    """

    if config.model.model == 'resnet18':
        loss = Resnet18Loss()

    # if config.loss.name == 'cross_entropy':
    #     loss = torch.nn.CrossEntropyLoss()
    # elif config.loss.name == 'mse':
    #     loss = torch.nn.MSELoss()
    else:
        raise ValueError(f"Loss {config.loss.name} not recognized.")

    return loss



class Resnet18Loss(torch.nn.Module):
    """
        Frankly we don't need this, we could have just as easily extract the loss_box_reg in the training loop.
    We're only doing this for completeness, in case we need to add more losses in the future, or simply to keep the code consistent.
    """
    def __init__(self) -> None:
        super(Resnet18Loss, self).__init__()

    def forward(self, outputs: dict) -> torch.Tensor:
        # return outputs['loss_classifier'], outputs['loss_box_reg'], outputs['loss_objectness'], outputs['loss_rpn_box_reg']
        return outputs['loss_box_reg']





