import os
import torch
from pathlib import Path
from typing import Any


def get_optimal_device_type() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_optimal_device():
    return torch.device(get_optimal_device_type())


def get_optimal_batch_size() -> int:
    free_mem, _ = get_memory_info()
    return 128 if free_mem >= 16.0 else 32


def set_default_device() -> torch.device:
    device = get_optimal_device()
    torch.set_default_device(device)
    return device


def set_default_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_default_optimizations():
    if torch.cuda.is_available():
        score = torch.cuda.get_device_capability()
        if score >= (8, 0):
            torch.set_float32_matmul_precision("high")
        else:
            torch.set_float32_matmul_precision("highest")


def get_parameters_number(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_memory_info(device: torch.device|None = None) -> tuple[float, float]:
    if device is None:
        device = get_optimal_device()
    free_mem, total_mem = torch.cuda.mem_get_info(device)
    return round(free_mem * 1e-9, 3), round(total_mem * 1e-9, 3)


def save_model(model: torch.nn.Module,
               target_dir: Path|str,
               model_name: str):
    """
    Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
                either ".pth" or ".pt" as the file extension.
    """
    if isinstance(target_dir, str):
        target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_path = target_dir / model_name
    torch.save(obj=model.state_dict(), f=model_path)


def export_model(model: torch.nn.Module,
                 target_dir: Path|str,
                 model_name: str,
                 args: tuple[Any, ...] = (),
                 version: int|None = None):
    """
    Export a PyTorch model to a target directory in ONNX format.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include ".onnx" as the file extension.
    args: A dummy inputs reflecting a production shape (e.g. (torch.zeros(1, 3, 224, 224),) if input is an RGB image)
    version: A version of ONNX opset to use (default: None)
    """
    if isinstance(target_dir, str):
        target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".onnx"), "model_name should end with '.onnx'"
    model_path = target_dir / model_name
    torch.onnx.export(model=model, args=args, f=model_path, opset_version=version)
