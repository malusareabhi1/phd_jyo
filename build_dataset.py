from pathlib import Path
from .utils import io_utils, split_utils

def run(cfg):
    annotations_dir = Path(cfg["paths"]["annotations_dir"])
    datasets_dir = Path(cfg["paths"]["datasets_dir"])
    datasets_dir.mkdir(parents=True, exist_ok=True)

    ann_df = io_utils.load_annotations(annotations_dir)
    train_df, val_df, test_df = split_utils.train_val_test_split(ann_df)
    io_utils.save_splits(train_df, val_df, test_df, datasets_dir)
