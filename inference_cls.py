from utils.paths import DATA_DIR
from utils.stratified_splits import build_train_val_datasets
from argparse import Namespace
from data_classes.datasets import USdatasetOmni
import torch, wandb
from sklearn.metrics import accuracy_score
from nets.cls_net import OmniClsCBAM
from utils.utils import organ_to_class_dict, multi_cls_labels_dict, generate_run_hash
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from train.cls_trainer import compute_metrics, FreezeBackboneCallback
from train.cls_trainer import get_sft_transforms
from safetensors.torch import load_file


def main():
    args = Namespace(
        wandb_entity="nmorelli-unimore",
        wandb_project="uusic_cls",
        wandb_run_name="cls_onpublic",  # Computed based on your logic
        dataset_type="both",
        dataset_size=256,
        use_ccl_crop=1,
        keep_aspect_ratio=1,
        rn_size="34",
        use_cbam=1,
        use_film=0,
        regr_bbox=1,
        cls_organ=0,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        learning_rate=0.0001,
        warmup_ratio=0.01,
        weight_decay=0.01,
        batch_size=64,
        num_workers=1,
        epochs=10,
        seed=42,
        debug=0,
        freeze_backbone=0,
        onpublic=1,
    )

    train_dataset, val_dataset = build_train_val_datasets(
        DATA_DIR, args, seed=42, get_sft_transforms_=get_sft_transforms
    )

    test_dataset = USdatasetOmni(
        DATA_DIR,
        "val_cls",
        transforms=get_sft_transforms(train=False),
        data_type="classification",
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
        resume=True,
        id="43ra7oz2",
    )
    model = OmniClsCBAM(
        resnet_size=args.rn_size,
        use_cbam=args.use_cbam,
        use_film=args.use_film,
        predict_bboxes=args.regr_bbox,
        mlp_organ=args.cls_organ,
    )

    run_hash = './loggings/d2c956748b76'
    output_dir = f"{run_hash}"
    print(f"Saving results to: {output_dir}")

    best_model = '/work/tesi_nmorelli/UUSIC_new/loggings/d2c956748b76/checkpoint-510/model.safetensors'
    state_dict = load_file(
        best_model
    )
    model.load_state_dict(state_dict)
    load_result = model.load_state_dict(state_dict)
    print(load_result)
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
    predictions = trainer.predict(test_dataset=test_dataset)
    print("Test results:", predictions.metrics)

if __name__ == "__main__":
    # import debugpy

    # debugpy.listen(("0.0.0.0", 5678))
    # print(">>> Debugger is listening on port 5678. Waiting for client to attach...")
    # debugpy.wait_for_client()
    # print(">>> Debugger attached. Resuming execution.")

    main()
