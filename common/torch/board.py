import os
import shutil
from pathlib import Path
from datetime import datetime
from torch.utils import tensorboard


def get_summary_writer(experiment_name: str,
                     model_name: str,
                     extra: str|None = None,
                     root_name: str = "runs",
                     reset: bool = True) -> tuple[str, tensorboard.SummaryWriter]:
    timestamp = datetime.now().strftime("%Y-%m-%d")
    logs_dir = os.path.join(root_name, timestamp, experiment_name, model_name)
    if extra is not None:
        logs_dir = os.path.join(logs_dir, extra)
    if reset is True and Path(logs_dir).exists():
        shutil.rmtree(logs_dir)
    return logs_dir, tensorboard.SummaryWriter(logs_dir)

