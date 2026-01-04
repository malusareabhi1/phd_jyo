import cv2
from pathlib import Path

def process_image(in_path: Path, out_path: Path, cfg: dict):
    img = cv2.imread(str(in_path))

    if cfg.get("resize", {}).get("enabled", False):
        img = cv2.resize(img, (cfg["resize"]["width"], cfg["resize"]["height"]))

    if cfg.get("grayscale", False):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if cfg.get("denoise", {}).get("enabled", False):
        k = cfg["denoise"]["ksize"]
        img = cv2.GaussianBlur(img, (k, k), 0)

    if cfg.get("threshold", {}).get("enabled", False):
        t = cfg["threshold"]
        img = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            t["block_size"],
            t["C"],
        )

    # TODO: add deskew/perspective correction here
    cv2.imwrite(str(out_path), img)
