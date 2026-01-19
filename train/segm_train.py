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
from torch.utils.data import Subset, ConcatDataset
from pathlib import Path
import pickle
from accelerate import Accelerator

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
        kmean_model = train_dataset.dataset.get_kmeans_model()
        # train_syn_dataset = USdatasetOmni(
        #     "/work/tesi_nmorelli/UUSIC_new/datasets/Synthetic_dataset_70_1.0_1.5_larger_filtered/pt_data",
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
            include_testicles=True,
            self_id = args.self_id,
            use_cluster_id = args.use_cluster_id,
            enc_type = args.enc_type,
            num_clusters = args.num_clusters,
            kmeans_model = kmean_model,
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
            include_testicles=True,
            self_id = args.self_id,
            enc_type = args.enc_type,
            use_cluster_id = args.use_cluster_id,
            num_clusters = args.num_clusters,
        )
        kmean_model = train_dataset.get_kmeans_model()

        # train_syn_dataset = USdatasetOmni(
        #     "/work/tesi_nmorelli/UUSIC_new/datasets/Synthetic_dataset_70_1.0_1.5_larger_filtered/pt_data",
        #     "train",
        #     transforms=get_sft_transforms(train=True),
        #     data_type=args.dataset_type,
        #     out_size=args.dataset_size,
        #     ccl_crop=args.use_ccl_crop,
        #     keep_aspect_ratio=args.keep_aspect_ratio,
        #     include_testicles=True,
        #     self_id = args.self_id,
        #     num_clusters = args.num_clusters,
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
            self_id = args.self_id,
            use_cluster_id = args.use_cluster_id,
            enc_type = args.enc_type,
            num_clusters = args.num_clusters,
            kmeans_model = kmean_model,
        )
        # train_dataset, val_dataset = build_train_val_datasets(
        #     "/work/tesi_nmorelli/UUSIC_new/datasets/Synthetic_dataset_70_1.0_1.5_larger_filtered/pt_data",
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
            self_id = bool(args.self_id),
            use_cluster_id = args.use_cluster_id,
            enc_type = args.enc_type,
            num_clusters = args.num_clusters,
            kmeans_model = kmean_model,
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
            # resume = True,
            # id = '9bv2or43'
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

    elif args.use_medsam_prompt:
        from segment_anything import sam_model_registry

        sam_model = sam_model_registry["vit_b"](checkpoint=MEDSAM_BASE_WEIGHTS)
        print(sum([p.numel() for p in sam_model.parameters()]))
        model = MedSAMPrompt(
            image_encoder=deepcopy(sam_model.image_encoder),
            mask_decoder=deepcopy(sam_model.mask_decoder),
            prompt_encoder=deepcopy(sam_model.prompt_encoder),
            predict_bboxes=True,
            freeze_image_encoder=args.freeze_image_encoder,
        )

    else:
        model = UNet2DFiLM(
            in_channels=3,
            num_classes=1,
            n_organs=args.num_clusters if bool(args.self_id) else len(organ_to_class_dict) ,
            size=32,
            depth=args.unet_depth,
            film_start=args.film_start,
            use_film=args.use_film,
            film_embed=args.film_embed,
            film_autoembed = bool(args.film_autoembed),
            distill = bool(args.distill)
        )
    # Generate custom hashed directory name
    run_hash = generate_run_hash(args)
    output_dir = f"{run_hash}"
    if kmean_model is not None:
        filepath = Path(f"{run_hash}/kmeans_model.pkl")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(kmean_model, f)

        print(f"KMeans model saved to {filepath}")
    print(f"Saving results to: {output_dir}")
    # for epochs
    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     num_train_epochs=args.epochs,
    #     per_device_train_batch_size=args.batch_size,
    #     per_device_eval_batch_size=args.batch_size,
    #     logging_dir="./logs",
    #     seed=args.seed,
    #     save_strategy="epoch",
    #     eval_strategy="epoch",
    #     load_best_model_at_end=True,
    #     metric_for_best_model="eval_loss",
    #     save_total_limit=2,
    #     report_to=["wandb"] if args.wandb_project else None,
    #     run_name=args.wandb_run_name,
    #     dataloader_num_workers=args.num_workers,
    #     logging_steps=10,
    #     log_level="info",
    #     eval_accumulation_steps=int(args.epochs),
    #     optim=args.optim,
    #     learning_rate=args.learning_rate,
    #     weight_decay=args.weight_decay,
    #     lr_scheduler_type=args.lr_scheduler_type,
    #     warmup_ratio=args.warmup_ratio,
    #     max_grad_norm=1.0,
    #     gradient_accumulation_steps=args.acc_grad,
    #     # fp16=True,
    #     # push_to_hub=False,
    # )
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.evaluate()

    predictions = trainer.predict(test_dataset=test_dataset)
    print("Test results:", predictions.metrics)
