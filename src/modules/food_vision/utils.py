from pathlib import Path
import torch

def save_model(model: torch.nn.Module,
               target_dir: Path,
               model_name: str):
    target_dir.mkdir(parents=True, exist_ok=True)
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), \
        "Model name should ends with '.pt' or '.pth'"
    model_save_path = target_dir/model_name
    print(f"Saving model to: {model_save_path}")
    torch.save(model.state_dict(), model_save_path)
