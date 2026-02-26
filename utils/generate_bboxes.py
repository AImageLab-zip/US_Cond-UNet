#!/usr/bin/env python3
"""
generate_bboxes.py – generate bounding-box prompts for *every* dataset
-----------------------------------------------------------------------------
Scans a *segmentation* root that contains many datasets of the form::

    <seg_root>/<DATASET>/masks/*.png (or .jpg / .tif …)

For **each** dataset it produces a JSON file ``<DATASET>_bboxes.json`` that
maps *mask* paths **relative to ``data_root``** to a dictionary with:
    * ``bbox_prompt`` --> exact bounding box
    * ``bbox_exp_prompt`` --> randomly *expanded* box (2-10 % larger per side)
    * ``bbox_red_prompt`` --> randomly *reduced* box (2-10 % tighter per side)

Usage
-----

    python generate_bboxes.py \
        --mask_ext png jpg \
        --seed 42

This creates e.g.::

    /work/…/segmentation/BUS-BRA/bboxes.json

-----
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


def bbox_from_mask(mask: np.ndarray) -> List[int]:
    """Compute [x_min, y_min, x_max, y_max] from a 2-D binary mask."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("Mask is empty – no foreground pixels found.")
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def jitter_bbox(
    bbox: List[int],
    img_w: int,
    img_h: int,
    expand: bool,
    r_min: float = 0.02,
    r_max: float = 0.10,
) -> List[int]:
    """Expand or contract *bbox* by a random factor in [r_min, r_max]."""
    x_min, y_min, x_max, y_max = bbox
    bw = x_max - x_min + 1
    bh = y_max - y_min + 1

    rx = random.uniform(r_min, r_max)
    ry = random.uniform(r_min, r_max)

    if expand:
        x_min = max(0, int(x_min - rx * bw))
        y_min = max(0, int(y_min - ry * bh))
        x_max = min(img_w - 1, int(x_max + rx * bw))
        y_max = min(img_h - 1, int(y_max + ry * bh))
    else:
        x_min = min(x_max - 1, int(x_min + rx * bw))
        y_min = min(y_max - 1, int(y_min + ry * bh))
        x_max = max(x_min + 1, int(x_max - rx * bw))
        y_max = max(y_min + 1, int(y_max - ry * bh))

    return [x_min, y_min, x_max, y_max]


def process_dataset(
    dataset_dir: Path,
    mask_exts: List[str],
    r_min: float,
    r_max: float,
) -> Dict[str, Dict[str, List[int]]]:
    """Return dict mapping RELATIVE mask paths -> bbox dicts for one dataset."""
    masks_dir = dataset_dir / "masks"
    if not masks_dir.is_dir():
        print(f"[WARN] {masks_dir} not found – skipped", file=sys.stderr)
        return {}

    patterns = [f"*.{ext.lower()}" for ext in mask_exts] + [f"*.{ext.upper()}" for ext in mask_exts]

    # Gather all mask files
    mask_files = []
    for pattern in patterns:
        mask_files.extend(masks_dir.rglob(pattern))

    if not mask_files:
        print(f"[WARN] no mask files in {masks_dir}", file=sys.stderr)
        return {}

    dataset_dict: Dict[str, Dict[str, List[int]]] = {}

    for mask_path in sorted(mask_files):
        try:
            mask = np.array(Image.open(mask_path).convert("L"))
            bbox = bbox_from_mask(mask)
            h, w = mask.shape
            bbox_exp = jitter_bbox(bbox, w, h, expand=True, r_min=r_min, r_max=r_max)
            bbox_red = jitter_bbox(bbox, w, h, expand=False, r_min=r_min, r_max=r_max)
        except Exception as e:
            print(f"[ERROR] {mask_path}: {e}", file=sys.stderr)
            continue

        dataset_dict[f"segmentation/{dataset_dir.name}/masks/{mask_path.name}"] = {
            "bbox_prompt": bbox,
            "bbox_exp_prompt": bbox_exp,
            "bbox_red_prompt": bbox_red,
        }

    return dataset_dict


def main():
    parser = argparse.ArgumentParser(description="Generate bbox prompts for every dataset in a segmentation root.")
    parser.add_argument("--dataset_dir", required=True, type=Path, help="Root directory containing dataset sub-folders with a masks/ dir inside.")
    parser.add_argument("--mask_ext", nargs="+", default=["png", "jpg", "jpeg", "tif", "tiff"], help="Mask file extensions (space-separated list).")
    parser.add_argument("--expansion_min", type=float, default=0.02)
    parser.add_argument("--expansion_max", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    dataset_dir: Path = args.dataset_dir


    if not dataset_dir.is_dir():
        return

    print(f"Processing dataset: {dataset_dir.name}")
    bboxes = process_dataset(
        dataset_dir=dataset_dir,
        mask_exts=args.mask_ext,
        r_min=args.expansion_min,
        r_max=args.expansion_max,
    )

    if bboxes:
        out_path = dataset_dir / f"bboxes.json"
        with open(out_path, "w") as f:
            json.dump(bboxes, f, indent=4)
        print(f"  → Saved {len(bboxes):,} entries to {out_path}")
    else:
        print(f"  (no boxes written)")

    files = []
    for file in (dataset_dir/"masks").iterdir():
        files.append(str(file))
    
    train_dim = int(len(files)*0.7)
    test_dim = int(len(files)*0.20)
    val_dim = len(files) - train_dim - test_dim

    random.shuffle(files)
    train_idxs = random.sample(range(len(files)), train_dim)
    val_idxs = random.sample(list(set(range(len(files))) - set(train_idxs)), val_dim)
    test_idxs = list(set(range(len(files))) - set(train_idxs) - set(val_idxs))

    with open(dataset_dir / "train.txt", "w") as f:
        for idx in train_idxs:
            f.write(files[idx] + "\n")
    with open(dataset_dir / "val.txt", "w") as f:
        for idx in val_idxs:
            f.write(files[idx] + "\n")
    with open(dataset_dir / "val_cls.txt", "w") as f:
        for idx in test_idxs:
            f.write(files[idx] + "\n")

    print(f"train.txt, val.txt, val_cls.txt created!!!")

    


if __name__ == "__main__":
    main()
