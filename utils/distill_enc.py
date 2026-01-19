from argparse import Namespace

from accelerate import Accelerator
import wandb
from data_classes.datasets import USdatasetOmni
from main_segm import parse_args
from nets.segm_net import MedSAM, UNet2DFiLM
from utils.stratified_splits import build_train_val_datasets, get_sft_transforms
from utils.utils import generate_run_hash, organ_to_class_dict
from segment_anything import sam_model_registry
from copy import deepcopy
from utils.paths import *
from torch import nn
from pathlib import Path
import pickle, torch
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from torchvision.transforms import v2
import torch.nn.functional as F


class DistillModule(nn.Module):
    def __init__(self, student, teacher, temperature=4.0):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        
        # Dynamically determine adaptation layer based on student depth
        student_depth = student.depth
        
        # Calculate student bottleneck output dimensions
        # depth=5: 2048 channels, 32x32
        # depth=4: 1024 channels, 64x64
        # depth=3: 512 channels, 128x128
        # etc.
        student_channels = 2048 // (2 ** (5 - student_depth))
        student_spatial = 32 * (2 ** (5 - student_depth))
        
        # Target: 256 channels, 64x64 (teacher output)
        target_channels = 256
        target_spatial = 64
        
        # Calculate upsampling factor
        spatial_factor = target_spatial // student_spatial
        
        if spatial_factor > 1:
            # Need upsampling
            self.adapt_layer = nn.ConvTranspose2d(
                student_channels, 
                target_channels, 
                kernel_size=3, 
                stride=spatial_factor, 
                padding=1, 
                output_padding=spatial_factor - 1
            )
        elif spatial_factor == 1:
            # Same spatial size, just adjust channels
            self.adapt_layer = nn.Conv2d(
                student_channels, 
                target_channels, 
                kernel_size=1
            )
        else:
            # Need downsampling
            downsample_factor = student_spatial // target_spatial
            self.adapt_layer = nn.Sequential(
                nn.Conv2d(
                    student_channels, 
                    target_channels, 
                    kernel_size=3, 
                    stride=downsample_factor, 
                    padding=1
                )
            )
        
        print(f"Student depth: {student_depth}")
        print(f"Student bottleneck: {student_channels} channels, {student_spatial}x{student_spatial}")
        print(f"Target (teacher): {target_channels} channels, {target_spatial}x{target_spatial}")
        print(f"Adaptation: {self.adapt_layer}")
        
        for p in self.teacher.parameters():
            p.requires_grad = False
    
    def forward(
        self,
        pixel_values,
        organ_id=None,
        labels=None,
        masks=None,
        bbox_coords=None,
        organ_id_metric=None,
    ):
        student_out = self.student.encode(pixel_values, organ_id)[0]
        student_out_adapted = self.adapt_layer(student_out)
        
        with torch.no_grad():
            self.teacher.eval()
            pixel_values_resized = v2.functional.resize(
                pixel_values, 1024, v2.InterpolationMode.BICUBIC
            )
            teacher_out = self.teacher.image_encoder(pixel_values_resized)
        
        # Ensure spatial dimensions match
        if student_out_adapted.shape != teacher_out.shape:
            # Interpolate if needed
            student_out_adapted = F.interpolate(
                student_out_adapted, 
                size=teacher_out.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Flatten spatial dimensions for KL divergence
        B, C, H, W = student_out_adapted.shape
        student_flat = student_out_adapted.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        teacher_flat = teacher_out.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        # Apply temperature scaling and softmax
        student_logits = F.log_softmax(student_flat / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_flat / self.temperature, dim=-1)
        
        # KL Divergence Loss
        distill_loss = F.kl_div(
            student_logits, 
            teacher_probs, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = distill_loss
        
        return {
            'loss': total_loss,
            'distill_loss': distill_loss,
            'student_features': student_out_adapted,
            'teacher_features': teacher_out
        }
    
def main(args: Namespace):
    student = UNet2DFiLM(
        in_channels=3,
        num_classes=1,
        n_organs=args.num_clusters if bool(args.self_id) else len(organ_to_class_dict),
        size=32,
        depth=args.unet_depth,
        film_start=args.film_start,
        use_film=args.use_film,
        film_embed=args.film_embed,
        film_autoembed=bool(args.film_autoembed),
        distill=bool(args.distill),
    )

    sam_model = sam_model_registry["vit_b"](checkpoint=MEDSAM_BASE_WEIGHTS)

    teacher = MedSAM(
        image_encoder=deepcopy(sam_model.image_encoder),
        mask_decoder=deepcopy(sam_model.mask_decoder),
        prompt_encoder=deepcopy(sam_model.prompt_encoder),
        predict_bboxes=True,
        freeze_image_encoder=0,
    )

    model = DistillModule(teacher=teacher, student=student)

    print(f"loaded distillation model")

    print("Loading public for train, private for test!")
    train_dataset, val_dataset = build_train_val_datasets(
        DATA_DIR, args, seed=args.seed, id_file_name="train_cls"
    )
    # kmean_model = train_dataset.dataset.get_kmeans_model()

    test_dataset = USdatasetOmni(
        DATA_DIR,
        "val_cls",
        transforms=get_sft_transforms(train=False),
        data_type=args.dataset_type,
        out_size=args.dataset_size,
        ccl_crop=args.use_ccl_crop,
        keep_aspect_ratio=args.keep_aspect_ratio,
        include_testicles=True,
        self_id = args.self_id,
        use_cluster_id = args.use_cluster_id,
        enc_type = args.enc_type,
        num_clusters = args.num_clusters,
        # kmeans_model = kmean_model,
    )
    print(
        f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}, Test dataset size: {len(test_dataset)}"
    )
    accelerator = Accelerator()

    if accelerator.is_main_process:
        wandb.login()
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=args,
        )
    # Generate custom hashed directory name
    run_hash = generate_run_hash(args)
    output_dir = f"{run_hash}"
    print(f"Saving results to: {output_dir}")
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
        dataloader_prefetch_factor=10,
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
        # bf16=True,
        # push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.evaluate()

    predictions = trainer.predict(test_dataset=test_dataset)
    print("Test results:", predictions.metrics)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    # import debugpy

    # debugpy.listen(("0.0.0.0", 5678))
    # print(">>> Debugger is listening on port 5678. Waiting for client to attach...")
    # debugpy.wait_for_client()
    # print(">>> Debugger attached. Resuming execution.")
    
    main(args)
