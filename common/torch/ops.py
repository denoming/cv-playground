import torch
from tqdm.auto import tqdm
from torch import nn
from torch.utils import data, tensorboard
from torch import optim
from torchmetrics.metric import Metric

def train_step(model: nn.Module,
               loader: data.DataLoader,
               optimizer: optim.Optimizer,
               criterion: nn.Module,
               metric: Metric,
               device: torch.device) -> tuple[float, float]:
    n_batches = len(loader)
    loss_avg, accu_avg = 0.0, 0.0
    metric.reset()
    model.train()
    for idx, (x, y) in tqdm(enumerate(loader), total=n_batches):
        x, y = x.to(device), y.to(device)
        y_logits = model(x)
        y_pred = torch.argmax(y_logits, dim=1)
        loss = criterion(y_logits, y)
        loss_avg += loss.item()
        metric.update(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_avg = loss_avg / n_batches
    accu_avg = metric.compute().item()
    return loss_avg, accu_avg


def test_step(model: nn.Module,
              loader: data.DataLoader,
              criterion: nn.Module,
              metric: Metric,
              device: torch.device) -> tuple[float, float]:
    n_batches = len(loader)
    loss_avg, accu_avg = 0.0, 0.0
    model.eval()
    with torch.inference_mode():
        for idx, (x, y) in tqdm(enumerate(loader), total=len(loader)):
            x, y = x.to(device), y.to(device)
            y_logits = model(x)
            y_pred = torch.argmax(y_logits, dim=1)
            loss = criterion(y_logits, y)
            loss_avg += loss.item()
            metric.update(y_pred, y)
    loss_avg = loss_avg / n_batches
    accu_avg = metric.compute().item()
    return loss_avg, accu_avg


def train(model: nn.Module,
          tr_dl: data.DataLoader,
          ts_dl: data.DataLoader,
          optimizer: torch.optim.Optimizer,
          criterion: nn.Module,
          metric: Metric,
          n_epochs: int,
          writer: tensorboard.SummaryWriter,
          device: torch.device,
          scheduler: optim.lr_scheduler.LRScheduler|None = None):
    for epoch in range(1, n_epochs+1):
        print(f"> Epoch: {epoch:02}")
        tr_loss, tr_accu = train_step(model, tr_dl, optimizer, criterion, metric, device)
        ts_loss, ts_accu = test_step(model, ts_dl, criterion, metric, device)
        if scheduler is not None:
            scheduler.step()
        writer.add_scalars(
            main_tag="Loss",
            tag_scalar_dict={"tr_loss": tr_loss, "ts_loss": ts_loss},
            global_step=epoch)
        writer.add_scalars(
            main_tag="Accuracy",
            tag_scalar_dict={"tr_accu": tr_accu, "ts_accu": ts_accu},
            global_step=epoch)
        print(f"Train L/A: {tr_loss:.3f}/{tr_accu:.3f} Test L/A: {ts_loss:.3f}/{ts_accu:.3f}")
        print()
    writer.close()