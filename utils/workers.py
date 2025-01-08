
from pathlib import Path
import time
import uuid

import torch
import torch.optim as optim
from tqdm import tqdm

from utils.utils import reset_random_seeds
from utils.logger import set_logger_paths, WandBLogger
from utils.data import get_data_loaders
from utils.training import create_optimizer, CustomMetrics, \
      train_one_epoch_resnet18, validate_one_epoch_resnet18, test_one_epoch_resnet18

from models.models import create_model
from models.losses import create_loss






def train(cfg):
    """
    Run the experiments for the Robotics Engineer Intern - AI Defect Detection - Flyability project. This method will set up the device,
    correct paths, initialize tracking, generate the dataset, train the model, and evaluate it.

    Args:
        cfg (dict): The configuration dictionary.
    """

    # --------------------------
    # Set up the device
    # --------------------------

    # Reproducibility
    gen_random_seed = reset_random_seeds(cfg['random_seed'])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Additional info when using cuda
    if device.type == 'cuda':
        print(f'Using {torch.cuda.get_device_name(0)}')
    else:
        print('Using CPU')
    
    # Set paths
    experiment_path, experiment_name = set_logger_paths(cfg)
    
    # WandB
    logger = WandBLogger(cfg, experiment_name)


    # --------------------------
    # Prepare the data loaders
    # --------------------------

    train_loader, val_loader, test_loader = get_data_loaders(cfg, gen_random_seed)


    # --------------------------
    # Initialize the model
    # --------------------------

    # Initialize a pre-trained model
    model = create_model(cfg).to(device)

    # load model, if it already exists
    if cfg.model.load_model_path != "":
        model.load_state_dict(torch.load(cfg.model.load_model_path, weights_only=True, map_location=device))

    model.to(device)

    loss_fn = create_loss(cfg)
    metrics = CustomMetrics(device=device)

    optimizer = create_optimizer(cfg, model) # torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.optimizer.decrease_every,
        gamma=1 / cfg.optimizer.lr_divisor,
    )
    num_epochs = cfg.training.epochs

    for epoch in range(num_epochs):
        """
            We will have the training and validation steps, encapsulated in a function, i.e. train_one_epoch and validate_one_epoch.
        By doing so, at each iteration of the training loop, we can have the training done, but at each validate_per_epoch, we
        will have the validation done. This way, we can have the model validated in a as many epochs as we wanted, in a single loop.
            Also epoch == 0 means 0 % cfg.training.validate_per_epoch == 0, so we will have the validation done at the first epoch, 
        so we just do epoch + 1 for validation.
        """
        print(f"Epoch {epoch + 1}")
        
        train_one_epoch_resnet18(
            model, train_loader, optimizer, loss_fn, metrics, device, epoch, logger
        )
        if epoch % cfg.training.validate_per_epoch == 0:
            validate_one_epoch_resnet18(
                model, val_loader, device, epoch, logger
            )
        lr_scheduler.step()
        # for _, batch in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch + 1}", leave=True, position=0)):
        #     images, image_names, targets = batch
        #     # Move to device
        #     images_input = [img.to(device) for img in images]
        #     targets_input = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #     # Forward pass
        #     outputs = model(images_input, targets_input)
        #     loss = sum(loss for loss in outputs.values())
        #     epoch_loss += loss.item()
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        # print(f"Epoch {epoch + 1} Loss: {epoch_loss}")
    
    # --------------------------
    # Test the model
    # --------------------------
    test_one_epoch_resnet18(
        model, test_loader, device, logger
    )

    # save the model
    if cfg.save_model:
        model_save_path = Path(experiment_path) / f"{experiment_name}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"\nTRAINING FINISHED, MODEL SAVED AS {model_save_path}!", flush=True)
    else:
        print("\nTRAINING FINISHED!", flush=True)

    x = 0

















