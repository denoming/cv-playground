import torch
from tqdm.auto import tqdm
from torch import nn
from torch.utils import data, tensorboard
from torch import optim
from torch import amp
from torchmetrics.metric import Metric
from typing import Callable


def train_step(model: Callable,
               loader: data.DataLoader,
               optimizer: optim.Optimizer,
               criterion: nn.Module,
               metric: Metric,
               scaler: amp.GradScaler,
               device: torch.device) -> tuple[float, float]:
    n_batches = len(loader)
    loss_avg, accu_avg = 0.0, 0.0
    metric.reset()
    for idx, (x, y) in tqdm(enumerate(loader), total=n_batches):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=scaler.is_enabled()):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_avg += loss.item()
        preds = torch.argmax(logits, dim=1)
        metric.update(preds, y)
    loss_avg = loss_avg / n_batches
    accu_avg = metric.compute().item()
    return loss_avg, accu_avg


def test_step(model: Callable,
              loader: data.DataLoader,
              criterion: nn.Module,
              metric: Metric,
              device: torch.device) -> tuple[float, float]:
    n_batches = len(loader)
    loss_avg, accu_avg = 0.0, 0.0
    with torch.inference_mode():
        for idx, (x, y) in tqdm(enumerate(loader), total=len(loader)):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            loss = criterion(logits, y)
            loss_avg += loss.item()
            metric.update(preds, y)
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
          use_amp: bool = True,
          use_compile: bool = True,
          scheduler: optim.lr_scheduler.LRScheduler|None = None):
    scaler = torch.amp.GradScaler(enabled=use_amp)
    xmodel = torch.compile(model) if use_compile else model
    for epoch in range(1, n_epochs+1):
        print(f"> Epoch: {epoch:02}")
        model.train(mode=True)
        tr_loss, tr_accu = train_step(xmodel, tr_dl, optimizer, criterion, metric, scaler, device)
        model.train(mode=False)
        ts_loss, ts_accu = test_step(xmodel, ts_dl, criterion, metric, device)
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