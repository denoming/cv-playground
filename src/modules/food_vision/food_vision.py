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
LEARNING_RATE = 0.001
N_EPOCHS = 10
N_HIDDEN_LAYERS = 32
BATCH_SIZE = 64
IMAGE_SIZE = (64,64)


def run(data_dir: Path, device: torch.device):
    tr_dl, ts_dl, classes = get_dataloaders(
        tr_dir=data_dir/"train",
        ts_dir=data_dir/"test",
        transform=v2.Compose([
            v2.Resize(size=IMAGE_SIZE),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float, scale=True)
        ]),
        batch_size=BATCH_SIZE,
        n_workers=2
    )

    model = TinyVgg(
        in_shape=3,
        out_shape=len(classes),
        hidden_units=N_HIDDEN_LAYERS
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    accuracy_fn = MulticlassAccuracy(num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    _ = train(model, tr_dl, ts_dl, optimizer, loss_fn, accuracy_fn, epochs=N_EPOCHS, device=device)

    save_model(model=model,
               target_dir=Path("files")/"foodvision",
               model_name="foodvision.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TinyVgg")
    parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--data_dir", required=True, type=Path, help="Path to image data")
    opts = parser.parse_args()

    opts.device = None
    if not opts.disable_cuda and torch.cuda.is_available():
        opts.device = torch.device("cuda")
    else:
        opts.device = torch.device("cpu")

    run(opts.data_dir, opts.device)






