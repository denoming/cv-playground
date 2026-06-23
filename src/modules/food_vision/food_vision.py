import argparse
from pathlib import Path

import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy
from torchvision.transforms import v2

from data import get_dataloaders
from model import TinyVgg
from ops import train
from utils import save_model

# Hyperparameters
N_HIDDEN_LAYERS = 32
# Image size
IMAGE_SIZE = (64,64)

# Enable CUDA graph optimization
torch.backends.cudnn.benchmark = True

def run(device: torch.device,
        path_to_data: Path,
        lr: float,
        epochs: int,
        batch_size: int):
    tr_dl, ts_dl, classes = get_dataloaders(
        tr_dir=path_to_data/"train",
        ts_dir=path_to_data/"test",
        tr_transform=v2.Compose([
            v2.Resize(size=IMAGE_SIZE),
            v2.TrivialAugmentWide(),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float, scale=True),
        ]),
        ts_transform=v2.Compose([
            v2.Resize(size=IMAGE_SIZE),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float, scale=True),
        ]),
        batch_size=batch_size,
        n_workers=2
    )

    model = TinyVgg(
        in_shape=3,
        out_shape=len(classes),
        hidden_units=N_HIDDEN_LAYERS
    ).to(device)

    estimator = MulticlassAccuracy(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    _ = train(model, tr_dl, ts_dl, optimizer, criterion, estimator, epochs=epochs, device=device)

    save_model(model=model,
               target_dir=Path("files")/"foodvision",
               model_name="foodvision.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TinyVgg")
    parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate (default: 0.001)")
    parser.add_argument("--epochs", default=10, type=int, help="The number of epochs to train (default: 10)")
    parser.add_argument("--batch_size", default=32, type=int, help="The size of batch (default: 32)")
    parser.add_argument("--path_to_data", required=True, type=Path, help="Path to data")
    opts = parser.parse_args()

    opts.device = None
    if not opts.disable_cuda and torch.cuda.is_available():
        opts.device = torch.device("cuda")
    else:
        opts.device = torch.device("cpu")

    run(opts.device, opts.path_to_data, opts.lr, opts.epochs, opts.batch_size)






