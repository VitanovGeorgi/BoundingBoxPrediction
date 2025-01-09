import torch


def get_customCNN():
    """
    Get the custom CNN model with the number of classes specified in the configuration file.
    This is just a modified VGG16 with in the middle removed.
    """
    model = torch.nn.Sequential(

        torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

        torch.nn.Linear(in_features=4096, out_features=128, bias=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.5, inplace=False),
        torch.nn.Linear(128, 4),
        torch.nn.Sigmoid()
    )
    return model