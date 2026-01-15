from argparse import Namespace
from collections import defaultdict
from safetensors.torch import load_file
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from data_classes.datasets import USdatasetOmni
from torchvision.transforms import InterpolationMode, v2
import torch, wandb, random
from sklearn.metrics import accuracy_score
from nets.cls_net import OmniClsCBAM
from nets.segm_net import UNet2DFiLM, MedSAM, MedSAMPrompt
from utils.utils import (
    organ_to_class_dict,
    multi_cls_labels_dict,
    class_to_organ_dict,
    get_sft_transforms,
    compute_dsc,
    compute_nsd,
    generate_run_hash,
)
import numpy as np
from utils.paths import *
from copy import deepcopy


def compute_metrics(eval_pred):
    logits, _ = eval_pred
    logits, masks, organ_ids = logits  # logits/masks shape B, 512, 512
    logits = v2.functional.resize(
        torch.from_numpy(logits),
        (masks.shape[-1], masks.shape[-1]),
        interpolation=InterpolationMode.NEAREST,
    ).numpy()
    pred_th = (torch.sigmoid(torch.from_numpy(logits)) > 0.7).float()
    pred_np = pred_th.cpu().numpy()
    masks = (
        (
            v2.functional.resize(
                torch.from_numpy(masks),
                (masks.shape[-1], masks.shape[-1]),
                interpolation=InterpolationMode.NEAREST,
            )
            > 0.5
        )
        .float()
        .numpy()
    )
    random.seed(42)  # set seed for reproducibility
    numbers = list(range(logits.shape[0]))
    sampled = random.sample(numbers, 30)
    overlays = []
    for s in sampled:
        overlays.append(
            mask_overlap_visualization(pred_th[s], torch.from_numpy(masks[s]))
        )

    organ_stats = defaultdict(lambda: {"dsc": [], "nsd": []})
    gt_np = masks
    for i, organ in enumerate(organ_ids):
        dsc = compute_dsc(gt_np[i], pred_np[i])
        nsd = compute_nsd(gt_np[i], pred_np[i], tolerance=1)
        organ = class_to_organ_dict[organ]
        organ_stats[organ]["dsc"].append(dsc)
        organ_stats[organ]["nsd"].append(nsd)

    wandb_metrics = {}
    for organ, lst in organ_stats.items():
        dsc_m = float(np.mean(lst["dsc"]))
        nsd_m = float(np.mean(lst["nsd"]))

        wandb_metrics[f"dsc_{organ}"] = dsc_m
        wandb_metrics[f"nsd_{organ}"] = nsd_m

    wandb_images = []
    for i, s in enumerate(sampled):
        wandb_images.append(wandb.Image(overlays[i], caption=f"overlap_{s}"))

    wandb.log({"overlays_eval": wandb_images}, commit=False)

    return wandb_metrics


def mask_overlap_visualization(pred: torch.Tensor, mask: torch.Tensor):
    """
    Create a color visualization tensor showing overlap between predicted mask and ground truth.
      - Green: correct prediction (TP)
      - Yellow: missed region (FN)
      - Red: false positive (FP)

    Args:
        mask (Tensor): shape (1, H, W) or (H, W), binary ground truth mask
        threshold (float): threshold for logits -> binary mask

    Returns:
        Tensor: RGB visualization tensor of shape (3, H, W), values in [0,1]
    """
    # Ensure proper shape - squeeze all dimensions of size 1
    pred = pred.squeeze()
    mask = mask.squeeze()

    # Ensure they are 2D

    # Logical masks
    tp = (pred == 1) & (mask == 1)  # True positives
    fn = (pred == 0) & (mask == 1)  # False negatives
    fp = (pred == 1) & (mask == 0)  # False positives

    # Create RGB channels
    h, w = pred.shape
    vis = torch.zeros(3, h, w)

    # Assign colors
    vis[0][tp] = 0  # R
    vis[1][tp] = 1  # G
    vis[2][tp] = 0  # B

    vis[0][fn] = 1  # R
    vis[1][fn] = 1  # G
    vis[2][fn] = 0  # B

    vis[0][fp] = 1  # R
    vis[1][fp] = 0  # G
    vis[2][fp] = 0  # B

    return vis


def train(args: Namespace):

    train_dataset = USdatasetOmni(
        DATA_DIR,
        "train",
        transforms=get_sft_transforms(train=True),
        data_type=args.dataset_type,
        out_size=args.dataset_size,
        ccl_crop=args.use_ccl_crop,
        keep_aspect_ratio=args.keep_aspect_ratio,
        self_norm=args.self_norm,
        include_testicles=True,
    )
    val_dataset = USdatasetOmni(
        DATA_DIR,
        "val",
        transforms=get_sft_transforms(train=False),
        data_type=args.dataset_type,
        out_size=args.dataset_size,
        ccl_crop=args.use_ccl_crop,
        keep_aspect_ratio=args.keep_aspect_ratio,
        self_norm=args.self_norm,
        include_testicles=True,
    )
    test_dataset = USdatasetOmni(
        DATA_DIR,
        "test",
        transforms=get_sft_transforms(train=False),
        data_type=args.dataset_type,
        out_size=args.dataset_size,
        ccl_crop=args.use_ccl_crop,
        keep_aspect_ratio=args.keep_aspect_ratio,
        self_norm=args.self_norm,
        include_testicles=True,
    )
    print(
        f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}, Test dataset size: {len(test_dataset)}"
    )
    wandb.login()
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=args,
        id = args.wandb_run_id,
        resume = True if args.wandb_run_id is not None else False
    )

    model = UNet2DFiLM(
        in_channels=3,
        num_classes=1,
        n_organs=len(organ_to_class_dict),
        size=32,
        depth=args.unet_depth,
        film_start=args.film_start,
    )
    state_dict = load_file(
        args.checkpoint_path
    )
    model.load_state_dict(state_dict)
    load_result = model.load_state_dict(state_dict)
    print(load_result)

    # Generate custom hashed directory name
    run_hash = generate_run_hash(args)
    output_dir = f"{run_hash}"
    print(f"Saving results to: {output_dir}")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_dir="./logs",
        seed=args.seed,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        report_to=["wandb"] if args.wandb_project else None,
        run_name=args.wandb_run_name,
        dataloader_num_workers=args.num_workers,
        logging_steps=10,
        log_level="info",
        eval_accumulation_steps=int(args.epochs),
        optim=args.optim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=1.0,
        gradient_accumulation_steps=args.acc_grad,
        # fp16=True,
        # push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Test dataset 0
    predictions = trainer.predict(test_dataset=test_dataset)
    print("Test results:", predictions.metrics)