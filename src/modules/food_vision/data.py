from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

def get_dataloaders(tr_dir: Path,
                    ts_dir: Path,
                    transform: T.Compose,
                    batch_size: int,
                    n_workers: int) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Creates training and testing DataLoaders.

    Takes in a training and testing dir paths and turns
    them into DataLoaders.

    Args:
        tr_dir: Path to training directory.
        ts_dir: Path to testing directory.
        transform: the list of transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        n_workers: An integer for number of workers per DataLoader.

    Returns:
        A tuple with training and testing data loaders with the list of classes.
    """
    tr_dataset = datasets.ImageFolder(tr_dir, transform=transform)
    ts_dataset = datasets.ImageFolder(ts_dir, transform=transform)
    classes = tr_dataset.classes
    tr_dl = DataLoader(
        tr_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
    )
    ts_dl = DataLoader(
        ts_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
    )
    return tr_dl, ts_dl, classes