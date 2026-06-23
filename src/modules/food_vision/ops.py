import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Callable
from tqdm.auto import tqdm

def train_step(model: nn.Module,
               loader: DataLoader,
               optimizer: optim.Optimizer,
               loss_fn: Callable,
               accuracy_fn: Callable,
               device: torch.device) -> tuple[float, float]:
    """
    Trains a PyTorch model for a single epoch.
    Args:
        model: A PyTorch model to be trained.
        loader: A DataLoader instance for the model to be trained on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn:  PyTorch loss function to minimize.
        accuracy_fn: PyTorch accuracy function to calculate accuracy.
        device: A target device to compute on.

    Returns:
    A tuple of training loss and training accuracy metrics.
    """
    loss_avg, accu_avg = 0, 0
    model.train()
    for idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        y_logits = model(x)
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
        loss = loss_fn(y_logits, y)
        accu = accuracy_fn(y_pred, y)
        loss_avg += loss.item()
        accu_avg += accu.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    n_batches = len(loader)
    loss_avg /= n_batches
    accu_avg /= n_batches
    return loss_avg, accu_avg

def test_step(model: nn.Module,
              loader: DataLoader,
              loss_fn: Callable,
              accuracy_fn: Callable,
              device: torch.device) -> tuple[float, float]:
    """
    Tests a PyTorch model for a single epoch.
    Args:
        model: A PyTorch model to be tested.
        loader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        accuracy_fn: PyTorch accuracy function to calculate accuracy.
        device: A target device to compute on.

    Returns:
    A tuple of testing loss and accuracy metrics.
    """
    loss_avg, accu_avg = 0, 0
    model.eval()
    with torch.inference_mode():
        for idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            y_logits = model(x)
            y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
            loss = loss_fn(y_logits, y)
            accu = accuracy_fn(y_pred, y)
            loss_avg += loss.item()
            accu_avg += accu.item()
    n_batches = len(loader)
    loss_avg /= n_batches
    accu_avg /= n_batches
    return loss_avg, accu_avg

def train(model: torch.nn.Module,
          tr_dl: torch.utils.data.DataLoader,
          ts_dl: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          accuracy_fn: Callable,
          epochs: int,
          device: torch.device) -> dict[str, list[float]]:
    """
    Trains and tests a PyTorch model.
    Args:
        model: A PyTorch model to be trained and tested.
        tr_dl: A DataLoader instance for the model to be trained on.
        ts_dl: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        accuracy_fn: PyTorch accuracy function to calculate accuracy.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute.

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    """
    results = {
        "tr_loss": [],
        "tr_acc": [],
        "ts_loss": [],
        "ts_acc": []
    }
    for epoch in tqdm(range(epochs)):
        tr_loss, tr_acc = train_step(model, tr_dl, optimizer, loss_fn, accuracy_fn, device)
        ts_loss, ts_acc = test_step(model, ts_dl, loss_fn, accuracy_fn, device)
        print(f"Epoch: {epoch+1:02} | Train L/A: {tr_loss:.4f}/{tr_acc:.4f} | Test L/A: {ts_loss:.4f}/{ts_acc:.4f}")
        results["tr_loss"].append(tr_loss)
        results["tr_acc"].append(tr_acc)
        results["ts_loss"].append(ts_loss)
        results["ts_acc"].append(ts_acc)
    return results