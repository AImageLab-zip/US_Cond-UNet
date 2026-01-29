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
        self._keys_to_ignore_on_save = None
        self._keys_to_ignore_on_load_missing = None
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
    Apply LoRA to the encoder part of UNet2DFiLM, including FiLM layers.
    
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
        # Add FiLM MLP layers (Linear layers inside FiLM blocks)
        elif isinstance(module, nn.Linear):
            target_modules.append(f"encoder.{name}")
    
    # Add bottleneck convolution and FiLM layers
    for name, module in model.bottleneck.named_modules():
        if isinstance(module, nn.Conv2d):
            target_modules.append(f"bottleneck.{name}")
        elif isinstance(module, nn.Linear):
            target_modules.append(f"bottleneck.{name}")
    
    print(f"\nApplying LoRA to {len(target_modules)} modules in encoder and bottleneck")
    print(f"Targeted module types: Conv2d, Linear (FiLM layers)")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=None,
    )
    
    # Apply LoRA (this will freeze ALL parameters by default)
    model = get_peft_model(model, lora_config)
    
    # CRITICAL: Unfreeze ALL decoder components
    decoder_params_count = 0
    for name, param in model.named_parameters():
        # Unfreeze decoder blocks
        if 'decoder' in name:
            param.requires_grad = True
            decoder_params_count += param.numel()
        # Unfreeze output layer
        elif 'out_layer' in name:
            param.requires_grad = True
            decoder_params_count += param.numel()
    
    print(f"\n=== Parameter Training Status ===")
    print(f"Decoder + out_layer parameters unfrozen: {decoder_params_count:,}")
    
    # Print detailed breakdown
    model.print_trainable_parameters()
    
    # Verify decoder is trainable
    print("\n=== Verification ===")
    decoder_trainable = sum(p.numel() for name, p in model.named_parameters() 
                           if p.requires_grad and ('decoder' in name or 'out_layer' in name))
    lora_trainable = sum(p.numel() for name, p in model.named_parameters() 
                        if p.requires_grad and 'lora' in name)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count FiLM LoRA params specifically
    film_lora_params = sum(p.numel() for name, p in model.named_parameters() 
                          if p.requires_grad and 'lora' in name and 'film' in name.lower())
    conv_lora_params = sum(p.numel() for name, p in model.named_parameters() 
                          if p.requires_grad and 'lora' in name and 'conv' in name.lower())
    
    print(f"Decoder trainable params: {decoder_trainable:,}")
    print(f"LoRA trainable params: {lora_trainable:,}")
    print(f"  - FiLM LoRA params: {film_lora_params:,}")
    print(f"  - Conv LoRA params: {conv_lora_params:,}")
    print(f"Total trainable params: {total_trainable:,}")
    print(f"Expected: {decoder_trainable + lora_trainable:,}")
    
    assert total_trainable == decoder_trainable + lora_trainable, \
        "Mismatch in trainable parameters! Some params may be incorrectly frozen."
    
    return model

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
    state_dict = load_file(
        "/work/phd_ultrasounds/UUSIC_new/checkpoints/medsam_unfreezed/model.safetensors"
    )
    load_result = teacher.load_state_dict(state_dict)
    print(f"Loaded MedSam teacher model and loaded weights:\n{load_result}")

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
