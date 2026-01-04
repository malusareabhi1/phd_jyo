from pathlib import Path
from .utils import io_utils

def run(cfg):
    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    # TODO: copy/rename input images into raw_dir using io_utils
