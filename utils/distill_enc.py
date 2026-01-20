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
from train.segm_train import compute_metrics
from peft import LoraConfig, get_peft_model, TaskType
from safetensors.torch import load_file

class DistillModule(nn.Module):
    def __init__(self, student, teacher, temperature=4.0):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        
        # Dynamically determine adaptation layer based on student depth
        student_depth = student.depth
        
        # Calculate student bottleneck output dimensions
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
            student_out_adapted = F.interpolate(
                student_out_adapted, 
                size=teacher_out.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Flatten spatial dimensions for KL divergence
        B, C, H, W = student_out_adapted.shape
        student_flat = student_out_adapted.view(B, C, -1).permute(0, 2, 1)
        teacher_flat = teacher_out.view(B, C, -1).permute(0, 2, 1)
        
        # Apply temperature scaling and softmax
        student_logits = F.log_softmax(student_flat / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_flat / self.temperature, dim=-1)
        
        # KL Divergence Loss
        distill_loss = F.kl_div(
            student_logits, 
            teacher_probs, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        total_loss = distill_loss
        
        return {
            'loss': total_loss,
            'distill_loss': distill_loss,
            'student_features': student_out_adapted,
            'teacher_features': teacher_out
        }


def apply_lora_to_encoder(model, lora_r=8, lora_alpha=16, lora_dropout=0.1):
    """
    Apply LoRA to the encoder part of UNet2DFiLM.
    
    Args:
        model: UNet2DFiLM model
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: Dropout probability for LoRA layers
    """
    # Identify encoder modules to apply LoRA to
    target_modules = []
    
    # Add encoder convolution layers
    for name, module in model.encoder.named_modules():
        if isinstance(module, nn.Conv2d):
            target_modules.append(f"encoder.{name}")
    
    # Add bottleneck convolution layers
    for name, module in model.bottleneck.named_modules():
        if isinstance(module, nn.Conv2d):
            target_modules.append(f"bottleneck.{name}")
    
    print(f"Applying LoRA to {len(target_modules)} modules in encoder and bottleneck")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=None,  # Don't save any modules explicitly
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


class UNet2DFiLMPEFTEncoder(UNet2DFiLM):
    """
    UNet2DFiLM with PEFT (LoRA) on encoder and bottleneck.
    
    This class extends UNet2DFiLM and applies LoRA adapters to the encoder
    and bottleneck, allowing parameter-efficient fine-tuning while keeping
    the decoder fully trainable.
    """
    
    def __init__(self, *args, lora_r=8, lora_alpha=16, lora_dropout=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self._lora_applied = False
    
    def apply_lora(self):
        """Apply LoRA to encoder and bottleneck."""
        if self._lora_applied:
            print("LoRA already applied, skipping...")
            return
        
        # Freeze encoder and bottleneck base parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.bottleneck.parameters():
            param.requires_grad = False
        
        # Apply LoRA manually to Conv2d layers
        self._apply_lora_to_module(self.encoder, "encoder")
        self._apply_lora_to_module(self.bottleneck, "bottleneck")
        
        self._lora_applied = True
        
        # Print parameter stats
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"LoRA applied - Trainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.2f}%)")
    
    def _apply_lora_to_module(self, module, prefix=""):
        """Recursively apply LoRA to Conv2d layers in a module."""
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                # Create LoRA adapter for this Conv2d
                lora_A = nn.Parameter(torch.randn(self.lora_r, child.in_channels, 1, 1) * 0.01)
                lora_B = nn.Parameter(torch.zeros(child.out_channels, self.lora_r, 1, 1))
                
                # Register as parameters
                full_name = f"{prefix}.{name}" if prefix else name
                setattr(module, f"{name}_lora_A", lora_A)
                setattr(module, f"{name}_lora_B", lora_B)
                
                print(f"Applied LoRA to {full_name}: {child.in_channels} -> {child.out_channels}")
            else:
                # Recursively apply to child modules
                self._apply_lora_to_module(child, f"{prefix}.{name}" if prefix else name)
    
    def forward(
        self,
        pixel_values,
        organ_id=None,
        labels=None,
        masks=None,
        bbox_coords=None,
        organ_id_metric=None,
    ):
        """Forward pass with LoRA adaptation."""
        # Apply LoRA adapters during forward pass
        x = pixel_values
        skip_connections = []
        
        # Encoder with LoRA
        for i, (down, film) in enumerate(zip(self.encoder, self.film_encoder)):
            if hasattr(self, f"encoder.{i}_lora_A"):
                # Apply LoRA adaptation
                lora_A = getattr(self, f"encoder.{i}_lora_A")
                lora_B = getattr(self, f"encoder.{i}_lora_B")
                base_out = down(x)
                lora_out = F.conv2d(F.conv2d(x, lora_A), lora_B)
                x = base_out + (self.lora_alpha / self.lora_r) * lora_out
            else:
                x = down(x)
            
            if organ_id is not None and film is not None:
                x = film(x, organ_id)
            skip_connections.append(x)
            x = F.max_pool2d(x, 2)
        
        # Bottleneck with LoRA
        if hasattr(self, "bottleneck_lora_A"):
            lora_A = getattr(self, "bottleneck_lora_A")
            lora_B = getattr(self, "bottleneck_lora_B")
            base_out = self.bottleneck(x)
            lora_out = F.conv2d(F.conv2d(x, lora_A), lora_B)
            x = base_out + (self.lora_alpha / self.lora_r) * lora_out
        else:
            x = self.bottleneck(x)
        
        if organ_id is not None and self.film_bottleneck is not None:
            x = self.film_bottleneck(x, organ_id)
        
        # Decoder (fully trainable)
        for up, film, skip in zip(self.decoder, self.film_decoder, reversed(skip_connections)):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = up(x)
            if organ_id is not None and film is not None:
                x = film(x, organ_id)
        
        logits = self.out_conv(x)
        
        # Calculate loss if masks provided
        loss = None
        if masks is not None:
            masks = masks.unsqueeze(1).float()
            loss = F.binary_cross_entropy_with_logits(logits, masks)
        
        return {
            'loss': loss,
            'logits': logits,
        }


def main(args: Namespace):  
    # ========================================
    # PHASE 1: Distillation Training
    # ========================================
    print("=" * 80)
    print("PHASE 1: Knowledge Distillation")
    print("=" * 80)
    
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

    # Apply LoRA to student encoder
    # Get LoRA hyperparameters from args or use defaults
    lora_r = getattr(args, 'lora_r', 128)
    lora_alpha = getattr(args, 'lora_alpha', 16)
    lora_dropout = getattr(args, 'lora_dropout', 0.1)
    
    # Stage 1: full distillation training (no PEFT)
    for param in student.parameters():
        param.requires_grad = True

    sam_model = sam_model_registry["vit_b"](checkpoint=MEDSAM_BASE_WEIGHTS)

    teacher = MedSAM(
        image_encoder=deepcopy(sam_model.image_encoder),
        mask_decoder=deepcopy(sam_model.mask_decoder),
        prompt_encoder=deepcopy(sam_model.prompt_encoder),
        predict_bboxes=True,
        freeze_image_encoder=0,
    )
    target_modules = [
            "qkv",      # Query, Key, Value projections in attention
            "proj",     # Output projection in attention
            "lin1",     # First linear layer in MLP
            "lin2",     # Second linear layer in MLP
        ]
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=None,  # Don't save any modules explicitly
    )
    teacher.image_encoder = get_peft_model(teacher.image_encoder, lora_config)


    state_dict = load_file(
        "/work/tesi_nmorelli/UUSIC_new/src/loggings/5d3d0b0842dc/checkpoint-4234/model.safetensors"
    )
    
    teacher.load_state_dict(state_dict)
    load_result = teacher.load_state_dict(state_dict)
    print(load_result)
    distill_model = DistillModule(teacher=teacher, student=student)

    print(f"Loaded distillation model with LoRA-enabled encoder")

    print("Loading public for train, private for test!")
    train_dataset, val_dataset = build_train_val_datasets(
        DATA_DIR, args, seed=args.seed, id_file_name="train_cls"
    )

    test_dataset = USdatasetOmni(
        DATA_DIR,
        "val_cls",
        transforms=get_sft_transforms(train=False),
        data_type=args.dataset_type,
        out_size=args.dataset_size,
        ccl_crop=args.use_ccl_crop,
        keep_aspect_ratio=args.keep_aspect_ratio,
        include_testicles=True,
        self_id=args.self_id,
        use_cluster_id=args.use_cluster_id,
        enc_type=args.enc_type,
        num_clusters=args.num_clusters,
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
            name=f"{args.wandb_run_name}_1",
            config=args,
        )
    
    # Generate custom hashed directory name
    run_hash = generate_run_hash(args)
    output_dir_phase1 = f"{run_hash}_1"
    print(f"Phase 1 - Saving results to: {output_dir_phase1}")
    
    training_args_phase1 = TrainingArguments(
        output_dir=output_dir_phase1,
        max_steps=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_dir="./logs",
        seed=args.seed,
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=int(args.epochs / 40),
        eval_steps=int(args.epochs / 40),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        report_to=["wandb"] if args.wandb_project else None,
        run_name=f"{args.wandb_run_name}_1",
        dataloader_num_workers=args.num_workers,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=6,
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
    )

    trainer_phase1 = Trainer(
        model=distill_model,
        args=training_args_phase1,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    print("\nStarting Phase 1 training...")
    trainer_phase1.train()
    trainer_phase1.evaluate()
    
    # Save the student model from phase 1 (including LoRA weights)
    phase1_student_path = Path(output_dir_phase1) / "student_encoder.pth"
    torch.save(student.state_dict(), phase1_student_path)
    print(f"\nPhase 1 complete. Saved student encoder with LoRA to: {phase1_student_path}")
    
    if accelerator.is_main_process:
        wandb.finish()
    
    # ========================================
    # PHASE 2: Segmentation Training (Decoder + LoRA)
    # ========================================
    print("\n" + "=" * 80)
    print("PHASE 2: Segmentation Training - Training Decoder + LoRA Encoder")
    print("=" * 80)
    args.dataset_type = 'segmentation'
    print("Loading public for train, private for test!")
    train_dataset, val_dataset = build_train_val_datasets(
        DATA_DIR, args, seed=args.seed, id_file_name="train_cls"
    )

    test_dataset = USdatasetOmni(
        DATA_DIR,
        "val_cls",
        transforms=get_sft_transforms(train=False),
        data_type=args.dataset_type,
        out_size=args.dataset_size,
        ccl_crop=args.use_ccl_crop,
        keep_aspect_ratio=args.keep_aspect_ratio,
        include_testicles=True,
        self_id=args.self_id,
        use_cluster_id=args.use_cluster_id,
        enc_type=args.enc_type,
        num_clusters=args.num_clusters,
    )
    # Create new student with same architecture
    student_phase2 = UNet2DFiLM(
        in_channels=3,
        num_classes=1,
        n_organs=args.num_clusters if bool(args.self_id) else len(organ_to_class_dict),
        size=32,
        depth=args.unet_depth,
        film_start=args.film_start,
        use_film=args.use_film,
        film_embed=args.film_embed,
        film_autoembed=bool(args.film_autoembed),
        distill=False,
    )
    
    # Load weights from phase 1 (full model) before applying PEFT
    student_phase2.load_state_dict(torch.load(phase1_student_path))
    print("Loaded full student weights from phase 1")
    
    # Stage 2: PEFT-only training
    student_phase2 = apply_lora_to_encoder(
        student_phase2,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    
    # Verify parameter counts
    trainable_params = sum(p.numel() for p in student_phase2.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in student_phase2.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    if accelerator.is_main_process:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=f"{args.wandb_run_name}_2",
            config=args,
        )
    
    output_dir_phase2 = f"{run_hash}_2"
    print(f"Phase 2 - Saving results to: {output_dir_phase2}")
    
    training_args_phase2 = TrainingArguments(
        output_dir=output_dir_phase2,
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
        run_name=f"{args.wandb_run_name}_2",
        dataloader_num_workers=args.num_workers,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=6,
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
    )

    trainer_phase2 = Trainer(
        model=student_phase2,
        args=training_args_phase2,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    print("\nStarting Phase 2 training with LoRA...")
    trainer_phase2.train()
    trainer_phase2.evaluate()

    print("\nRunning final test evaluation...")
    predictions = trainer_phase2.predict(test_dataset=test_dataset)
    print("Test results:", predictions.metrics)
    
    # Save final model
    final_model_path = Path(output_dir_phase2) / "final_model_lora.pth"
    torch.save(student_phase2.state_dict(), final_model_path)
    print(f"\nPhase 2 complete. Saved final model with LoRA to: {final_model_path}")
    
    if accelerator.is_main_process:
        wandb.finish()
    
    print("\n" + "=" * 80)
    print("TWO-PHASE TRAINING WITH PEFT (LoRA) COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    # import debugpy

    # debugpy.listen(("0.0.0.0", 5678))
    # print(">>> Debugger is listening on port 5678. Waiting for client to attach...")
    # debugpy.wait_for_client()
    # print(">>> Debugger attached. Resuming execution.")
    main(args)
