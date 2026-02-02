from argparse import Namespace
from collections import defaultdict
from copy import deepcopy
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from data_classes.datasets import USdatasetOmni
from torchvision.transforms import InterpolationMode, v2
import torch, wandb, random
from sklearn.metrics import accuracy_score
from nets.cls_net import OmniClsCBAM
from nets.segm_net import UNet2DFiLM, MedSAM, MedSAMPrompt
from nets.unet_attn import UNet2DAttn
from utils.paths import DATA_DIR
from utils.utils import organ_to_class_dict, multi_cls_labels_dict, generate_run_hash
import numpy as np
from utils.utils import (
    get_sft_transforms,
    compute_dsc,
    class_to_organ_dict,
    compute_nsd,
    mask_overlap_visualization,
)
from utils.stratified_splits import build_train_val_datasets
from utils.paths import *
from torch.utils.data import DataLoader, Subset, ConcatDataset
from pathlib import Path
import pickle
from accelerate import Accelerator
from utils.sampler import BalancedHierarchicalSampler
from transformers import TrainerCallback

def compute_metrics(eval_pred):
    logits, _ = eval_pred
    # logits, masks, organ_ids = logits  # logits/masks shape B, 512, 512
    logits, masks, organ_ids, organ_id_metric = logits  # logits/masks shape B, 512, 512

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
    for i, organ in enumerate(organ_id_metric):
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

    if wandb.run is not None:
        wandb_images = []
        for i, s in enumerate(sampled):
            wandb_images.append(wandb.Image(overlays[i], caption=f"overlap_{s}"))

        wandb.log({"overlays_eval": wandb_images}, commit=False)

    return wandb_metrics


def train(args: Namespace):
    if args.onpublic:
        print("Loading public for train, private for test!")
        train_dataset, val_dataset = build_train_val_datasets(
            DATA_DIR, args, seed=args.seed, id_file_name="train_cls"
        )
        # train_syn_dataset = USdatasetOmni(
        #     "/work/phd_ultrasounds/UUSIC_new/datasets/Synthetic_dataset_70_1.0_1.5_larger_filtered/pt_data",
        #     "train",
        #     transforms=get_sft_transforms(train=True),
        #     data_type=args.dataset_type,
        #     out_size=args.dataset_size,
        #     ccl_crop=args.use_ccl_crop,
        #     keep_aspect_ratio=args.keep_aspect_ratio,
        #     include_testicles=True,
        # )
        # print(f"Synthetic dataset initialized with {len(train_syn_dataset.items)} images")
        # train_dataset = ConcatDataset([train_dataset, train_syn_dataset])

        test_dataset = USdatasetOmni(
            DATA_DIR,
            "val_cls",
            transforms=get_sft_transforms(train=False),
            data_type=args.dataset_type,
            out_size=args.dataset_size,
            ccl_crop=args.use_ccl_crop,
            keep_aspect_ratio=args.keep_aspect_ratio,
            skip_dataset=args.val_skip_dataset,
            id_dropout=0.0,
            teacher_cache_dir = '/work/phd_ultrasounds/UUSIC_new/datasets/medsam_cache/train'

        )
    else:
        train_dataset = USdatasetOmni(
            DATA_DIR,
            "train",
            transforms=get_sft_transforms(train=True),
            data_type=args.dataset_type,
            out_size=args.dataset_size,
            ccl_crop=args.use_ccl_crop,
            keep_aspect_ratio=args.keep_aspect_ratio,
            self_norm=args.self_norm,
            id_dropout=args.id_dropout,
        )

        # train_syn_dataset = USdatasetOmni(
        #     "/work/phd_ultrasounds/UUSIC_new/datasets/Synthetic_dataset_70_1.0_1.5_larger_filtered/pt_data",
        #     "train",
        #     transforms=get_sft_transforms(train=True),
        #     data_type=args.dataset_type,
        #     out_size=args.dataset_size,
        #     ccl_crop=args.use_ccl_crop,
        #     keep_aspect_ratio=args.keep_aspect_ratio,
        #     include_testicles=True,
        # )
        # print(
        #     f"Synthetic dataset initialized with {len(train_syn_dataset.items)} images"
        # )
        # train_dataset = ConcatDataset([train_dataset, train_syn_dataset])

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
            id_dropout=0.0,
        )
        # train_dataset, val_dataset = build_train_val_datasets(
        #     "/work/phd_ultrasounds/UUSIC_new/datasets/Synthetic_dataset_70_1.0_1.5_larger_filtered/pt_data",
        #     args,
        #     seed=args.seed,
        #     id_file_name="train",
        # )
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
            id_dropout=0.0,
        )

    print(
        f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}, Test dataset size: {len(test_dataset)}"
    )
    train_sampler = BalancedHierarchicalSampler(
        dataset=train_dataset,
        batch_size=args.batch_size,
        steps_per_epoch=int(args.epochs / 50),
        seed=args.seed,
    )

    if args.use_medsam:
        from segment_anything import sam_model_registry

        sam_model = sam_model_registry["vit_b"](checkpoint=MEDSAM_BASE_WEIGHTS)

        model = MedSAM(
            image_encoder=deepcopy(sam_model.image_encoder),
            mask_decoder=deepcopy(sam_model.mask_decoder),
            prompt_encoder=deepcopy(sam_model.prompt_encoder),
            predict_bboxes=True,
            freeze_image_encoder=0,
        )
        model.cuda()

    elif args.use_medsam_prompt:
        from segment_anything import sam_model_registry

        sam_model = sam_model_registry["vit_b"](checkpoint=MEDSAM_BASE_WEIGHTS)
        print(sum([p.numel() for p in sam_model.parameters()]))
        model = MedSAMPrompt(
            image_encoder=deepcopy(sam_model.image_encoder),
            mask_decoder=deepcopy(sam_model.mask_decoder),
            prompt_encoder=deepcopy(sam_model.prompt_encoder),
            predict_bboxes=True,
            freeze_image_encoder=0,
            n_organs=len(organ_to_class_dict),
        )
        model.cuda()

    elif args.unet_attn:
        model = UNet2DAttn(
            in_channels=3,
            num_classes=1,
            n_organs=len(organ_to_class_dict),
            size=32,
            depth=args.unet_depth,
            attn_start=args.film_start,  # Start attention from first level
            use_attn=args.use_film,  # Enable attention
            img_size=512,  # Input image size
            patch_size=8,  # 16×16 patches → 256 patches total
            emb_dim=768,  # Embedding dimension
            n_heads=8,  # Number of attention heads
            # n_transformer_layers = 12,
            distill=bool(args.distill),
            use_dwt=args.use_dwt,
            wavelet=args.wavelet,
            dwt_bands=args.dwt_bands,
        )
    else:
        model = UNet2DFiLM(
            in_channels=3,
            num_classes=1,
            n_organs=len(organ_to_class_dict) ,
            size=32,
            depth=args.unet_depth,
            film_start=args.film_start,
            use_film=args.use_film,
            distill = bool(args.distill)
        )


    # Generate custom hashed directory name
    if args.resume == None:
        run_hash = generate_run_hash(args)
    else:
        run_hash = f"./loggings/{args.resume}"
    
    run_id = str(run_hash).split('/')[-1]
    output_dir = f"{run_hash}"
    print(f"Saving results to: {output_dir}")
    accelerator = Accelerator()

    if accelerator.is_main_process:
        wandb.login()
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=args,
            resume = 'allow' if args.resume != None else 'never',
            id = run_id,
        )
    # for steps
    training_args = TrainingArguments(
        output_dir=output_dir,
        # num_train_epochs=args.epochs,
        max_steps=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_dir="./logs",
        seed=args.seed,
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=int(args.epochs / 50),
        eval_steps=int(args.epochs / 50),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        report_to=["wandb"] if args.wandb_project else None,
        run_name=args.wandb_run_name,
        dataloader_num_workers=args.num_workers,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=20,
        logging_steps=10,
        log_level="info",
        eval_accumulation_steps=100,
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

    trainer = CustomTrainerWithSampler(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        train_sampler=train_sampler,
        callbacks=[DistillScheduleCallback(args.distill)]
    )
    # trainer = CustomTrainerWithSampler(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset,
    #     eval_dataset=dataset,
    #     compute_metrics=compute_metrics,
    # )
    trainer.train(resume_from_checkpoint=(args.resume != None) )
    trainer.evaluate()

    predictions = trainer.predict(test_dataset=test_dataset)
    print("Test results:", predictions.metrics)


class CustomTrainerWithSampler(Trainer):
    """
    Custom Trainer that uses BalancedHierarchicalSampler for training.
    """

    def __init__(self, *args, train_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_sampler = train_sampler

    def get_train_dataloader(self):
        """
        Override to use our custom batch sampler.
        """
        if self.train_sampler is None:
            # Fallback to default behavior
            return super().get_train_dataloader()

        # Create DataLoader with our batch sampler
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=(
                self.args.dataloader_prefetch_factor
                if self.args.dataloader_num_workers > 0
                else None
            ),
        )


class DistillScheduleCallback(TrainerCallback):
    def __init__(self, stop_step):
        self.stop_step = stop_step
        print(f"Distillation will only be applied for the first {stop_step} steps")
    def on_step_begin(self, args, state, control, **kwargs):
        if self.stop_step is None:
            return
        model = kwargs["model"]
        if hasattr(model, "distill"):
            if model.distill != (state.global_step < self.stop_step):
                print("----------------DISTILLATION STOPPED-----------------")

            model.distill = state.global_step < self.stop_step
        
