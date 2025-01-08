
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor











def custom_colate(data):
    """ 
        We would need this if we had more than one object to be detected in the image, of the same class.
    Since we don't, we don't need this, i.e. no need of a custom collate function.
    """
    return data




def get_FastRCNN(num_classes : int = 2) -> torch.nn.Module:
    """
    Get the FastRCNN model with the number of classes specified in the configuration file.
    https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html

    Args:
        cfg (DictConfig): The configuration dictionary.

    Returns:
        torch.nn.Module: The FastRCNN model.
    """
    # Load a pre-trained model for classification and return
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model



