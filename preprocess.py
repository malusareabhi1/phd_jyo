from pathlib import Path
from .utils import image_utils

def run(cfg):
    raw_dir = Path(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    for img_path in raw_dir.rglob("*.jpg"):
        out_path = processed_dir / img_path.name
        image_utils.process_image(img_path, out_path, cfg["preprocess"])
