import torch


def get_customCNN():
    """
    Get the custom CNN model with the number of classes specified in the configuration file.
    """
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(64 * 28 * 28, 128), # 28 * 28 is the size of the image after two maxpooling
        torch.nn.ReLU(),
        torch.nn.Linear(128, 4),
        torch.nn.Sigmoid()
    )
    return model