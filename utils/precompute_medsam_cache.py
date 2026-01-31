#!/usr/bin/env python3
"""
Precompute MedSAM teacher embeddings and logits for distillation.

This script mirrors the distillation pathway in nets/segm_net.py:
- image_embedding = MedSAM.image_encoder(resized pixel_values)
- low_res_masks = MedSAM.mask_decoder(...)
- mid_res_masks = resize(low_res_masks, 512)

Note: If your training uses random augmentations, caching with eval transforms
will not match the augmented student inputs. Consider disabling random aug
when using cached teachers.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from data_classes.datasets import USdatasetOmni
from nets.segm_net import MedSAM
from utils.paths import DATA_DIR, MEDSAM_BASE_WEIGHTS
from utils.utils import get_sft_transforms


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.base[idx], idx


def _resolve_output_path(out_dir: Path, data_dir: Path, image_path: str) -> Path:
    image_path = Path(image_path)
    try:
        rel = image_path.relative_to(data_dir)
    except ValueError:
        rel = Path(image_path.parent.name) / image_path.name
    return (out_dir / rel).with_suffix(".pt")


def _load_teacher(base_ckpt: str, teacher_ckpt: Optional[str]) -> MedSAM:
    from segment_anything import sam_model_registry

    sam_model = sam_model_registry["vit_b"](checkpoint=base_ckpt)
    teacher = MedSAM(
        image_encoder=deepcopy(sam_model.image_encoder),
        mask_decoder=deepcopy(sam_model.mask_decoder),
        prompt_encoder=deepcopy(sam_model.prompt_encoder),
        predict_bboxes=True,
        freeze_image_encoder=0,
    )

    if teacher_ckpt:
        if teacher_ckpt.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(teacher_ckpt)
        else:
            state_dict = torch.load(teacher_ckpt, map_location="cpu")
        load_result = teacher.load_state_dict(state_dict, strict=False)
        print(f"Loaded teacher checkpoint: {teacher_ckpt}\n{load_result}")

    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    return teacher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute MedSAM teacher cache")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--dataset-type", type=str, default="both")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ccl-crop", action="store_true")
    parser.add_argument("--keep-aspect-ratio", action="store_true", default=True)
    parser.add_argument("--no-keep-aspect-ratio", action="store_false", dest="keep_aspect_ratio")
    parser.add_argument("--self-norm", action="store_true")
    parser.add_argument("--medsam-base", type=str, default=MEDSAM_BASE_WEIGHTS)
    parser.add_argument("--teacher-checkpoint", type=str, default=None)
    parser.add_argument("--dtype", type=str, choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--use-train-transforms", action="store_true")
    parser.add_argument("--save-low-res", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transforms = get_sft_transforms(train=args.use_train_transforms)

    dataset = USdatasetOmni(
        args.data_dir,
        args.split,
        transforms=transforms,
        data_type=args.dataset_type,
        ccl_crop=args.ccl_crop,
        out_size=512,
        keep_aspect_ratio=args.keep_aspect_ratio,
        self_norm=args.self_norm,
        id_dropout=0.0,
    )

    loader = DataLoader(
        IndexedDataset(dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    teacher = _load_teacher(args.medsam_base, args.teacher_checkpoint).to(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    processed = 0
    with torch.no_grad():
        for batch, indices in tqdm(loader, desc="Caching MedSAM", unit="batch"):
            pixel_values = batch["pixel_values"].to(args.device, non_blocking=True)
            up_pixel_values = v2.functional.resize(
                pixel_values, 1024, v2.InterpolationMode.BICUBIC
            )
            image_embedding = teacher.image_encoder(up_pixel_values)
            image_pe = teacher.prompt_encoder.get_dense_pe()
            low_res_masks, _ = teacher.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=teacher.learned_sparse_embeddings,
                dense_prompt_embeddings=teacher.learned_dense_embeddings,
                multimask_output=False,
            )
            mid_res_masks = v2.functional.resize(
                low_res_masks, 512, v2.InterpolationMode.BICUBIC
            )

            image_embedding = image_embedding.to(dtype).cpu()
            mid_res_masks = mid_res_masks.to(dtype).cpu()
            low_res_masks = low_res_masks.to(dtype).cpu()

            for i, idx in enumerate(indices):
                item = dataset.items[idx]
                out_path = _resolve_output_path(out_dir, Path(args.data_dir), item["image_path"])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.exists() and not args.overwrite:
                    continue

                payload = {
                    "image_embedding": image_embedding[i],
                    "mid_res_masks": mid_res_masks[i],
                }
                if args.save_low_res:
                    payload["low_res_masks"] = low_res_masks[i]

                torch.save(payload, out_path)
                processed += 1
                if args.max_samples is not None and processed >= args.max_samples:
                    print(f"Reached max samples: {args.max_samples}")
                    return


if __name__ == "__main__":
    main()
