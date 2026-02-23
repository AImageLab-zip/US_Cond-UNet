from argparse import Namespace
from copy import deepcopy
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from data_classes.datasets import USdatasetOmni
from torchvision.transforms import InterpolationMode, v2
import torch, wandb, random
from torchmetrics.functional.segmentation import dice_score
from nets.segm_net import UNet2DFiLM, MedSAM, MedSAMPrompt, nnUnetWrapper
from nets.unet_attn import UNet2DAttn
from utils.paths import DATA_DIR
from utils.utils import organ_to_class_dict, multi_cls_labels_dict, generate_run_hash
import numpy as np
from utils.utils import (
    get_sft_transforms,
    class_to_organ_dict,
    compute_nsd,
    mask_overlap_visualization,
)
from utils.stratified_splits import build_train_val_datasets
from utils.paths import *
from torch.utils.data import DataLoader, Subset, ConcatDataset
from pathlib import Path
from accelerate import Accelerator
from utils.sampler import BalancedHierarchicalSampler
from transformers import TrainerCallback


def _to_tensor_on_device(arr, device):
    """Convert numpy array to torch tensor on `device` preserving dtype."""
    if isinstance(arr, torch.Tensor):
        return arr.to(device)
    return torch.as_tensor(arr, device=device)

def make_compute_metrics(accelerator=None):
    """
    Returns a compute_metrics(eval_pred) function suitable for HF Trainer.
    If `accelerator` is provided (an accelerate.Accelerator instance), predictions
    will be gathered from all processes before metric computation.
    """
    def compute_metrics(eval_pred):
        # Unpack eval_pred exactly like your original function expects
        # eval_pred may already be global (numpy arrays) or local per-rank. Handle both.
        logits_out, _ = eval_pred
        # In your original code logits_out was itself a tuple:
        # logits, masks, organ_ids, organ_id_metric = logits_out
        # So handle that shape:
        if isinstance(logits_out, (list, tuple)) and len(logits_out) >= 4:
            logits_np, masks_np, organ_ids_np, organ_id_metric_np = logits_out[:4]
        else:
            raise ValueError("Unexpected eval_pred contents; expected (logits, masks, organ_ids, organ_id_metric).")

        # If using accelerator, gather tensors from all processes
        if accelerator is not None:
            device = accelerator.device
            # convert to tensors on device
            logits_t = _to_tensor_on_device(logits_np, device)
            masks_t = _to_tensor_on_device(masks_np, device)
            organ_ids_t = _to_tensor_on_device(organ_ids_np, device)
            organ_id_metric_t = _to_tensor_on_device(organ_id_metric_np, device)

            # Use gather_for_metrics if available, else gather
            gather_fn = getattr(accelerator, "gather_for_metrics", None) or getattr(accelerator, "gather", accelerator.gather)

            # gather will block until all ranks reach here
            try:
                logits_all = gather_fn(logits_t)
                masks_all = gather_fn(masks_t)
                organ_ids_all = gather_fn(organ_ids_t)
                organ_id_metric_all = gather_fn(organ_id_metric_t)
            except Exception:
                # fallback to plain gather
                logits_all = accelerator.gather(logits_t)
                masks_all = accelerator.gather(masks_t)
                organ_ids_all = accelerator.gather(organ_ids_t)
                organ_id_metric_all = accelerator.gather(organ_id_metric_t)

            # move to cpu numpy for the heavy numpy logic that follows
            logits = logits_all.detach().cpu().numpy()
            masks = masks_all.detach().cpu().numpy()
            organ_ids = organ_ids_all.detach().cpu().numpy()
            organ_id_metric = organ_id_metric_all.detach().cpu().numpy()
        else:
            # single-process (or Trainer already gathered): keep numpy arrays
            logits = np.asarray(logits_np)
            masks = np.asarray(masks_np)
            organ_ids = np.asarray(organ_ids_np)
            organ_id_metric = np.asarray(organ_id_metric_np)

        # --- now the original metric logic, slightly adapted ---
        batch_size = logits.shape[0]
        chunk_size = min(32, batch_size)

        dsc_scores_list = []
        nsd_scores_list = []
        overlays = []

        random.seed(42)
        numbers = list(range(batch_size))
        sampled = random.sample(numbers, min(30, batch_size))
        sampled_set = set(sampled)

        for start_idx in range(0, batch_size, chunk_size):
            end_idx = min(start_idx + chunk_size, batch_size)

            # Convert chunk to torch on CPU/GPU as needed (we'll use CPU for resizing if you prefer)
            logits_chunk = torch.from_numpy(logits[start_idx:end_idx])
            masks_chunk = torch.from_numpy(masks[start_idx:end_idx])

            # Resize (use same API you used originally)
            logits_resized = v2.functional.resize(
                logits_chunk,
                (masks_chunk.shape[-1], masks_chunk.shape[-1]),
                interpolation=InterpolationMode.NEAREST,
            )
            pred_th_chunk = (torch.sigmoid(logits_resized) > 0.7).float()

            masks_resized = v2.functional.resize(
                masks_chunk,
                (masks_chunk.shape[-1], masks_chunk.shape[-1]),
                interpolation=InterpolationMode.NEAREST,
            )
            masks_chunk = (masks_resized > 0.5).float()

            # Generate overlays for sampled indices in this chunk
            for local_idx in range(end_idx - start_idx):
                global_idx = start_idx + local_idx
                if global_idx in sampled_set:
                    # ensure overlay op works with cpu tensors / numpy arrays per your implementation
                    overlay = mask_overlap_visualization(pred_th_chunk[local_idx], masks_chunk[local_idx])
                    overlays.append((global_idx, overlay))

            # Compute metrics for chunk
            pred_idx_chunk = pred_th_chunk.squeeze(1).to(torch.long)
            gt_idx_chunk = masks_chunk.squeeze(1).to(torch.long)

            # DSC (your dice_score returns per-sample array)
            dsc_chunk = dice_score(
                pred_idx_chunk,
                gt_idx_chunk,
                num_classes=2,
                include_background=False,
                average="micro",
                input_format="index",
                aggregation_level="samplewise",
            )
            dsc_chunk = torch.nan_to_num(dsc_chunk, nan=0.0).cpu().numpy()
            dsc_scores_list.append(dsc_chunk)

            # NSD - compute per-sample
            pred_np_chunk = pred_idx_chunk.cpu().numpy()
            gt_np_chunk = gt_idx_chunk.cpu().numpy()

            nsd_chunk = np.array(
                [compute_nsd(gt_np_chunk[i], pred_np_chunk[i], tolerance=1)
                 for i in range(pred_np_chunk.shape[0])],
                dtype=np.float32,
            )
            nsd_scores_list.append(nsd_chunk)

            # Clear memory
            del logits_chunk, masks_chunk, logits_resized, masks_resized
            del pred_th_chunk, pred_idx_chunk, gt_idx_chunk, pred_np_chunk, gt_np_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Concatenate results
        dsc_scores = np.concatenate(dsc_scores_list) if dsc_scores_list else np.array([], dtype=np.float32)
        nsd_scores = np.concatenate(nsd_scores_list) if nsd_scores_list else np.array([], dtype=np.float32)

        # Compute per-organ metrics
        organ_ids_arr = np.asarray(organ_id_metric)  # your code used organ_id_metric for per-sample organ
        wandb_metrics = {}
        dsc_values_for_mean = []

        for organ_id in np.unique(organ_ids_arr):
            organ_mask = organ_ids_arr == organ_id
            organ_name = class_to_organ_dict[int(organ_id)]

            # guard empty selection
            if organ_mask.sum() == 0:
                dsc_mean = float(0.0)
                nsd_mean = float(0.0)
            else:
                dsc_mean = float(dsc_scores[organ_mask].mean())
                nsd_mean = float(nsd_scores[organ_mask].mean())

            wandb_metrics[f"dsc_{organ_name}"] = dsc_mean
            wandb_metrics[f"nsd_{organ_name}"] = nsd_mean

            if "unknown" not in organ_name.lower():
                dsc_values_for_mean.append(dsc_mean)

        if dsc_values_for_mean:
            wandb_metrics["dsc_mean"] = float(np.mean(dsc_values_for_mean))

        # Log overlays only on main process (or always if no accelerator)
        log_overlays = (accelerator.is_main_process if accelerator is not None else True)
        if wandb.run is not None and log_overlays:
            overlays.sort(key=lambda x: x[0])
            wandb_images = [wandb.Image(overlay, caption=f"overlap_{idx}") for idx, overlay in overlays]
            wandb.log({"overlays_eval": wandb_images}, commit=False)

        return wandb_metrics

    return compute_metrics


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
            transforms=get_sft_transforms(train=False, size =int(args.dataset_size)),
            data_type=args.dataset_type,
            out_size=args.dataset_size,
            ccl_crop=args.use_ccl_crop,
            keep_aspect_ratio=args.keep_aspect_ratio,
            self_norm=args.self_norm,
            skip_dataset=args.val_skip_dataset,
            id_dropout=0.0,
        )
    else:
        if int(getattr(args, "fold", 0)) > 0:
            train_dataset, val_dataset = build_train_val_datasets(
                DATA_DIR, args, seed=args.seed, id_file_name="train"
            )
        else:
            train_dataset = USdatasetOmni(
                DATA_DIR,
                "train",
                transforms=get_sft_transforms(train=True, size =int(args.dataset_size)),
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

        if int(getattr(args, "fold", 0)) <= 0:
            val_dataset = USdatasetOmni(
                DATA_DIR,
                "val",
                transforms=get_sft_transforms(train=False, size =int(args.dataset_size)),
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
            transforms=get_sft_transforms(train=False, size =int(args.dataset_size) ),
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

    elif args.use_nnunet:
        model = nnUnetWrapper()
    elif args.unet_attn:
        model = UNet2DAttn(
            in_channels=3,
            num_classes=1,
            n_organs=len(organ_to_class_dict),
            size=args.unet_size,
            depth=args.unet_depth,
            attn_start=args.film_start,  # Start attention from first level
            use_attn=args.use_film,  # Enable attention
            img_size=args.dataset_size,  # Input image size
            patch_size=8,  # 16×16 patches → 256 patches total
            emb_dim=768,  # Embedding dimension
            n_heads=8,  # Number of attention heads
            distill=bool(args.distill),
            distill_unet=bool(args.distill_unet),
            use_dwt=args.use_dwt,
            wavelet=args.wavelet,
            dwt_bands=args.dwt_bands,
            use_shape=args.use_shape,
            shape_res=args.shape_res,
        )
    else:
        model = UNet2DFiLM(
            in_channels=3,
            num_classes=1,
            n_organs=len(organ_to_class_dict) ,
            size=args.unet_size,
            depth=args.unet_depth,
            film_start=args.film_start,
            use_film=args.use_film,
            distill = bool(args.distill),
            distill_unet = bool(args.distill_unet),
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

    distill_steps = args.distill if bool(args.distill) else args.distill_unet
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=val_dataset,
    #     compute_metrics=compute_metrics,
    #     callbacks=[DistillScheduleCallback(distill_steps)]
    # )
    trainer = CustomTrainerWithSampler(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=make_compute_metrics(accelerator),
        train_sampler=train_sampler,
        callbacks=[DistillScheduleCallback(distill_steps)]
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
        if hasattr(model, "distill") and model.distill:
            if model.distill != (state.global_step < self.stop_step):
                print("----------------DISTILLATION STOPPED-----------------")

            model.distill = state.global_step < self.stop_step
        elif hasattr(model, "distill_unet") and model.distill_unet:
            if model.distill_unet != (state.global_step < self.stop_step):
                print("----------------DISTILLATION STOPPED-----------------")

            model.distill_unet = state.global_step < self.stop_step
