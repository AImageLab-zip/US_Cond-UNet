from argparse import Namespace

from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from data_classes.datasets import USdatasetOmni
from torchvision.transforms import v2
import torch, wandb
from sklearn.metrics import accuracy_score
from nets.cls_net import OmniClsCBAM
from utils.paths import DATA_DIR
from utils.utils import (
    organ_to_class_dict,
    multi_cls_labels_dict,
    generate_run_hash
)
from utils.stratified_splits import build_train_val_datasets
from transformers import TrainerCallback, TrainerState, TrainerControl
from torch.utils.data import Subset, ConcatDataset


def get_sft_transforms(train: bool):
    """Get image transforms for training/validation"""
    if train:
        return v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomApply(
                    [
                        v2.RandomAffine(degrees=15, shear=10),
                    ],
                    p=0.5,
                ),
                v2.RandomErasing(p=0.4, scale=(0.05, 0.3), ratio=(0.3, 3.3)),
                v2.ColorJitter(brightness=0.4, contrast=0.4),
                v2.ToDtype(torch.float32, scale=False),
                v2.Normalize(
                    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
                ),
            ]
        )
    else:
        return v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=False),
                v2.Normalize(
                    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
                ),
            ]
        )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits, organ_ids = logits
    dataset_names = list(organ_to_class_dict.keys())
    predictions = []
    prediction_per_dataset = {}
    label_per_dataset = {}
    for logit, label, organ_label in zip(logits, labels, organ_ids):
        if label != -100:
            dataset_name = dataset_names[organ_label]
            labels_set = multi_cls_labels_dict[dataset_name]
            prediction = logit[labels_set].argmax() + labels_set[0]
            predictions.append(prediction)
            if dataset_name in prediction_per_dataset.keys():
                prediction_per_dataset[dataset_name].append(prediction)
            else:
                prediction_per_dataset[dataset_name] = [prediction]

            if dataset_name in label_per_dataset.keys():
                label_per_dataset[dataset_name].append(label)
            else:
                label_per_dataset[dataset_name] = [label]

        else:
            predictions.append(-100)
    predictions = torch.Tensor(predictions)
    mask = labels != -100
    acc = accuracy_score(labels[mask], predictions[mask])
    return_dict = {}
    for k in label_per_dataset.keys():
        dataset_labels = torch.Tensor(label_per_dataset[k])
        dataset_preds = torch.Tensor(prediction_per_dataset[k])
        return_dict[f"acc_{k}"] = accuracy_score(dataset_labels, dataset_preds)

    return_dict["acc_mean"] = torch.Tensor(list(return_dict.values())).mean()
    return_dict["acc_all"] = acc
    return return_dict


class FreezeBackboneCallback(TrainerCallback):
    """
    A callback to unfreeze the model's backbone after the first training epoch.
    """

    def __init__(self, n_epochs=1.0):
        super().__init__()
        self.n_epochs = n_epochs

    def on_epoch_end(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        # Check if the first epoch has just finished (epoch index starts at 0, so epoch 1.0 means the first one is done)
        if state.epoch >= self.n_epochs and state.epoch < (self.n_epochs + 1.0):
            print(
                "--- First epoch complete. Unfreezing backbone for remaining training. ---"
            )

            # The model is available via kwargs['model']
            model = kwargs["model"]

            # Call the unfreeze function
            model.unfreeze_backbone()
            # Re-initialize the optimizer to include the newly trainable parameters.
            # This is CRITICAL because the original optimizer was only initialized
            # with the *trainable* parameters present before trainer.train() was called.
            control.should_reinitialize_optimizer = True

            # You might want to adjust the learning rate here if your LR scheduler
            # only applies to the first phase, but typically you let the scheduler handle it.


def train(args: Namespace):
    if args.onpublic:
        train_dataset, val_dataset = build_train_val_datasets(DATA_DIR, args, seed=42, get_sft_transforms_ = get_sft_transforms)

        train_syn_dataset = USdatasetOmni(
            "/work/tesi_nmorelli/UUSIC_new/datasets/Synthetic_dataset_70_1.0_1.5_larger_filtered/pt_data",
            "train",
            transforms=get_sft_transforms(train=True),
            data_type=args.dataset_type,
            out_size=args.dataset_size,
            ccl_crop=args.use_ccl_crop,
            keep_aspect_ratio=args.keep_aspect_ratio,
            include_testicles=True,
        )
        print(f"Synthetic dataset initialized with {len(train_syn_dataset.items)} images")
        train_dataset = ConcatDataset([train_dataset, train_syn_dataset])

        test_dataset = USdatasetOmni(
            DATA_DIR,
            "val_cls",
            transforms=get_sft_transforms(train=False),
            data_type=args.dataset_type,
            out_size=args.dataset_size,
            ccl_crop=args.use_ccl_crop,
            keep_aspect_ratio=args.keep_aspect_ratio,
            include_testicles=True
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
        )
        val_dataset = USdatasetOmni(
            DATA_DIR,
            "val",
            transforms=get_sft_transforms(train=False),
            data_type=args.dataset_type,
            out_size=args.dataset_size,
            ccl_crop=args.use_ccl_crop,
            keep_aspect_ratio=args.keep_aspect_ratio,
        )
        test_dataset = USdatasetOmni(
            DATA_DIR,
            "test",
            transforms=get_sft_transforms(train=False),
            data_type=args.dataset_type,
            out_size=args.dataset_size,
            ccl_crop=args.use_ccl_crop,
            keep_aspect_ratio=args.keep_aspect_ratio,
        )
    print(
        f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}, Test dataset size: {len(test_dataset)}"
    )
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=args,
    )
    model = OmniClsCBAM(
        resnet_size=args.rn_size,
        use_cbam=args.use_cbam,
        use_film=args.use_film,
        predict_bboxes=args.regr_bbox,
        mlp_organ=args.cls_organ,
    )

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
        report_to=["wandb"] if args.wandb_project else None,
        run_name=args.wandb_run_name,
        dataloader_num_workers=args.num_workers,
        logging_steps=10,
        save_total_limit=2,
        log_level="info",
        eval_accumulation_steps=int(args.epochs),
        optim=args.optim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=1.0,
        # fp16=True,
        # push_to_hub=False,
    )

    if args.freeze_backbone:
        model.freeze_backbone()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=(
            [FreezeBackboneCallback(args.freeze_backbone)]
            if args.freeze_backbone
            else None
        ),
    )
    trainer.train()
    trainer.evaluate()

    predictions = trainer.predict(test_dataset=test_dataset)
    print("Test results:", predictions.metrics)
