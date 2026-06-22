import torch
from torch import nn
from torchvision.transforms import v2
from torchmetrics.classification import MulticlassAccuracy
from pathlib import Path
from food_vision.model import TinyVgg
from food_vision.data import get_dataloaders
from food_vision.ops import train
from food_vision.utils import save_model
from common import CV_DATASETS_DIR


# Hyperparameters
LEARNING_RATE = 0.001
N_EPOCHS = 10
N_HIDDEN_LAYERS = 32
BATCH_SIZE = 64
IMAGE_SIZE = (64,64)

# Dataset dirs
DATASET_PATH = CV_DATASETS_DIR/"food"/"pizza_steak_sushi"
TR_DATASET_DIR = DATASET_PATH / "train"
TS_DATASET_DIR = DATASET_PATH / "test"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_dl, ts_dl, classes = get_dataloaders(
        tr_dir=TR_DATASET_DIR,
        ts_dir=TS_DATASET_DIR,
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
    run()






