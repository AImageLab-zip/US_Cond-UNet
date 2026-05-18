import argparse
import os
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import Trainer
from transformers.training_args import TrainingArguments

from data_classes.datasets import USdatasetOmni
from nets.segm_net import MedSAM, MedSAMPrompt, UNet2DFiLM
from nets.unet_attn import UNet2DAttn
from train.segm_train import make_compute_metrics
from utils.paths import DATA_DIR, MEDSAM_BASE_WEIGHTS
from utils.utils import get_sft_transforms, organ_to_class_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference for segmentation checkpoints trained with train/segm_train.py."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Checkpoint file or checkpoint directory. If directory, model.safetensors is preferred.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./loggings/inference_tmp",
        help="Temporary output directory required by HuggingFace Trainer.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("SLURM_JOB_CPUS_PER_NODE", "8")),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument("--eval-accumulation-steps", type=int, default=100)

    parser.add_argument("--dataset-type", type=str, default="segmentation")
    parser.add_argument("--dataset-size", type=int, default=512)
    parser.add_argument("--use-ccl-crop", type=int, default=0)
    parser.add_argument("--keep-aspect-ratio", type=int, default=0)
    parser.add_argument("--self-norm", type=int, default=0)
    parser.add_argument("--onpublic", type=int, default=0)
    parser.add_argument("--val-skip-dataset", type=str, default="")
    parser.add_argument(
        "--selfaug",
        type=int,
        default=0,
        help="Set to 1 only if you intentionally want self-augmentation test-time pairs.",
    )

    parser.add_argument("--film-start", type=int, default=0)
    parser.add_argument("--use-film", type=int, default=1)
    parser.add_argument("--unet-depth", type=int, default=5)
    parser.add_argument("--unet-size", type=int, default=32)
    parser.add_argument("--unet-attn", type=int, default=0)
    parser.add_argument("--use-medsam", type=int, default=0)
    parser.add_argument("--use-medsam-prompt", type=int, default=0)
    parser.add_argument("--freeze-image-encoder", type=int, default=1)
    parser.add_argument("--use-dwt", type=int, default=0)
    parser.add_argument("--wavelet", type=str, default="haar")
    parser.add_argument(
        "--dwt-bands",
        nargs="+",
        default=["LL", "LH", "HL", "HH"],
        help="DWT subbands to use (any combination of: LL LH HL HH).",
    )
    parser.add_argument("--use-shape", type=int, default=0)
    parser.add_argument("--shape-res", type=int, default=32)
    parser.add_argument(
        "--strict-load",
        type=int,
        default=0,
        help="Use strict state-dict loading (1) or allow missing/unexpected keys (0).",
    )
    args = parser.parse_args()

    if args.use_medsam and args.use_medsam_prompt:
        raise ValueError("Only one of --use-medsam or --use-medsam-prompt can be enabled.")

    allowed = {"LL", "LH", "HL", "HH"}
    raw_bands = []
    for band in args.dwt_bands:
        for part in band.split(","):
            part = part.strip()
            if part:
                raw_bands.append(part)
    normalized = []
    seen = set()
    for band in raw_bands:
        band = band.upper()
        if band in seen:
            continue
        if band not in allowed:
            raise ValueError(f"Invalid DWT band '{band}'. Valid bands: {sorted(allowed)}")
        normalized.append(band)
        seen.add(band)
    args.dwt_bands = normalized
    return args


def resolve_checkpoint_file(path: str) -> Path:
    ckpt_path = Path(path)
    if ckpt_path.is_file():
        return ckpt_path
    if not ckpt_path.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")

    def _find_model_file(folder: Path):
        candidates = [
            folder / "model.safetensors",
            folder / "pytorch_model.bin",
            folder / "model.bin",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    direct_file = _find_model_file(ckpt_path)
    if direct_file is not None:
        return direct_file

    checkpoint_dirs = sorted(
        [
            d
            for d in ckpt_path.glob("checkpoint-*")
            if d.is_dir() and d.name.split("-")[-1].isdigit()
        ],
        key=lambda d: int(d.name.split("-")[-1]),
        reverse=True,
    )
    for ckpt_dir in checkpoint_dirs:
        ckpt_file = _find_model_file(ckpt_dir)
        if ckpt_file is not None:
            return ckpt_file

    raise FileNotFoundError(
        f"No model file found in checkpoint directory {ckpt_path}. "
        "Expected model.safetensors/pytorch_model.bin/model.bin directly in that "
        "folder or in a checkpoint-* subfolder."
    )


def build_model(args):
    if args.use_medsam:
        from copy import deepcopy
        from segment_anything import sam_model_registry

        sam_model = sam_model_registry["vit_b"](checkpoint=MEDSAM_BASE_WEIGHTS)
        return MedSAM(
            image_encoder=deepcopy(sam_model.image_encoder),
            mask_decoder=deepcopy(sam_model.mask_decoder),
            prompt_encoder=deepcopy(sam_model.prompt_encoder),
            predict_bboxes=True,
            freeze_image_encoder=bool(args.freeze_image_encoder),
        )

    if args.use_medsam_prompt:
        from copy import deepcopy
        from segment_anything import sam_model_registry

        sam_model = sam_model_registry["vit_b"](checkpoint=MEDSAM_BASE_WEIGHTS)
        return MedSAMPrompt(
            image_encoder=deepcopy(sam_model.image_encoder),
            mask_decoder=deepcopy(sam_model.mask_decoder),
            prompt_encoder=deepcopy(sam_model.prompt_encoder),
            predict_bboxes=True,
            freeze_image_encoder=bool(args.freeze_image_encoder),
            n_organs=len(organ_to_class_dict),
        )

    if args.unet_attn:
        return UNet2DAttn(
            in_channels=3,
            num_classes=1,
            n_organs=len(organ_to_class_dict),
            size=args.unet_size,
            depth=args.unet_depth,
            attn_start=args.film_start,
            use_attn=args.use_film,
            img_size=args.dataset_size,
            patch_size=8,
            emb_dim=768,
            n_heads=8,
            distill=False,
            distill_unet=False,
            use_dwt=args.use_dwt,
            wavelet=args.wavelet,
            dwt_bands=args.dwt_bands,
            use_shape=args.use_shape,
            shape_res=args.shape_res,
            use_selfaug=args.selfaug,
        )

    return UNet2DFiLM(
        in_channels=3,
        num_classes=1,
        n_organs=len(organ_to_class_dict),
        size=args.unet_size,
        depth=args.unet_depth,
        film_start=args.film_start,
        use_film=args.use_film,
        distill=False,
        distill_unet=False,
        use_selfaug=args.selfaug,
    )


def load_checkpoint_weights(model, checkpoint_file: Path, strict: bool):
    if checkpoint_file.suffix == ".safetensors":
        state_dict = load_file(str(checkpoint_file))
    else:
        raw = torch.load(str(checkpoint_file), map_location="cpu")
        state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw

    state_dict_wo_distill = {k:v for k, v in state_dict.items() if 'distill' not in k}
    load_result = model.load_state_dict(state_dict_wo_distill, strict=strict)
    print(f"Loaded checkpoint: {checkpoint_file}")
    print(f"Load result: {load_result}")


def build_test_dataset(args):
    split = "val_cls" if args.onpublic else "test"
    test_dataset = USdatasetOmni(
        DATA_DIR,
        split,
        transforms=get_sft_transforms(train=False, size=int(args.dataset_size)),
        data_type=args.dataset_type,
        out_size=args.dataset_size,
        ccl_crop=args.use_ccl_crop,
        keep_aspect_ratio=args.keep_aspect_ratio,
        self_norm=args.self_norm,
        skip_dataset=args.val_skip_dataset,
        use_selfaug=args.selfaug,
    )
    print(f"Test split: {split}")
    print(f"Test dataset size: {len(test_dataset)}")
    return test_dataset


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model = build_model(args)
    checkpoint_file = resolve_checkpoint_file(args.checkpoint_path)
    load_checkpoint_weights(model, checkpoint_file, strict=bool(args.strict_load))

    test_dataset = build_test_dataset(args)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=not args.cpu,
        dataloader_persistent_workers=args.num_workers > 0,
        dataloader_prefetch_factor=20 if args.num_workers > 0 else None,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        eval_accumulation_steps=args.eval_accumulation_steps,
        no_cuda=args.cpu,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=make_compute_metrics(accelerator=None),
    )

    predictions = trainer.predict(test_dataset=test_dataset)
    print("Test results:")
    for key in sorted(predictions.metrics.keys()):
        print(f"{key}: {predictions.metrics[key]}")


if __name__ == "__main__":
    main()
