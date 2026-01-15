import argparse, multiprocessing, torch, os


def parse_args():
    """Parses command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train model configuration.")

    # -----------------------------------------------
    # |               WANDB Config                  |
    # -----------------------------------------------
    
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default='nmorelli-unimore',
        help="Weights & Biases entity name",
    )    
    parser.add_argument(
        "--wandb-project",
        type=str,
        default='uusic_segm',
        help="Weights & Biases project name. If set, enables W&B logging.",
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None, help="Weights & Biases run name."
    )
    
    parser.add_argument(
        "--wandb-run-id", type=str, default=None, help="Weights & Biases run id in case you want to resume."
    )
    # -----------------------------------------------
    # |               DATASET Config                |
    # -----------------------------------------------
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="segmentation",
        choices=["both", "segmentation", "classification"],
        help="what data to load, available segmnetation and classification",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=512,
        help="dimension of the images used by the dataset",
    )
    parser.add_argument(
        "--use-ccl-crop", help="if use ccl cropping", type=int, default=0
    )
    parser.add_argument(
        "--keep-aspect-ratio",
        help="if use keep aspect ration when resizing",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--self-norm",
        help="if use self norm",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--testicle-split",
        type=str,
        default="",
    )    
    parser.add_argument(
        "--self-id",
        type=int,
        default=0,
    )    
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=10,
    )
    # -----------------------------------------------
    # |                 SEGM Config                 |
    # -----------------------------------------------
    parser.add_argument(
        "--film-start",
        help="where to start add films, if bigger then depth it wont be used",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--use-film",
        type=int,
        default=1,
    ) 
    parser.add_argument(
        "--unet-depth", help="number of stage in the uner", type=int, default=5
    )
    parser.add_argument(
        "--use-medsam",
        type=int,
        default=0,
    )     
    parser.add_argument(
        "--use-medsam-prompt",
        type=int,
        default=0,
    )    
    parser.add_argument(
        "--freeze-image-encoder",
        type=int,
        default=1,
    )    
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
    )    
    parser.add_argument(
        "--film-embed",
        type=int,
        default=64,
    )    
    parser.add_argument(
        "--film-autoembed",
        type=bool,
        default=1,
    )
    
    # -----------------------------------------------
    # |         Optim & Scheduler Config            |
    # -----------------------------------------------
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        choices=["adamw_torch", "adamw_apex_fused", "adafactor", "sgd"],
        help="The optimizer to use. 'adamw_hf' is generally recommended for transformers.",
    )

    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="linear",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
        help="The learning rate scheduler type to use.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="lr",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.06,
        help="Ratio of total training steps used for linear warmup.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for the optimizer.",
    )

    # -----------------------------------------------
    # |               Training Config               |
    # -----------------------------------------------
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Input batch size for training."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("SLURM_JOB_CPUS_PER_NODE", "8")),
        help="Number of workers for data loading.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )

    parser.add_argument("--debug", help="flag for debugging", type=int, default=0)
    parser.add_argument(
        "--freeze-backbone",
        type=int,
        default=0,
        help="numbers of epoch for warmup heads",
    )    
    parser.add_argument(
        "--acc-grad",
        type=int,
        default=1,
        help="acc batches for gradient",
    )
    parser.add_argument(
        "--onpublic",
        type=int,
        default=0,
        help="to use if want to train on public data and test on private",
    )
    parser.add_argument(
        "--sft",
        type=int,
        default=0,
        help="to use if want to finetune on any dataset of your choice",
    )
    parser.add_argument(
        "--testicle-sft",
        type=int,
        default=0,
        help="to use if want to finetune on testicle dataset",
    )
    parser.add_argument(
        "--use-syn",
        type=int,
        default=0,
        help="to use if want to finetune on synthetic testicle dataset",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    if args.wandb_project and not args.wandb_run_name:
        # Construct a descriptive run name from the most important swept args
        name_parts = [
            f"RN{args.rn_size}",
            f"CBAM{args.use_cbam}",
            f"FILM{args.use_film}",
            f"BBOX{args.regr_bbox}",
            f"LR{args.learning_rate:.0e}",  # Format as scientific notation
            f"WR{args.warmup_ratio}",
            f"FB{args.freeze_backbone}",
        ]
        args.wandb_run_name = "_".join(name_parts).replace(".", "p")
        print(f"Constructed W&B Run Name: {args.wandb_run_name}")
    if args.debug :
        import debugpy

        debugpy.listen(("0.0.0.0", 5678))
        print(">>> Debugger is listening on port 5678. Waiting for client to attach...")
        debugpy.wait_for_client()
        print(">>> Debugger attached. Resuming execution.")

    if args.testicle_sft:
        from train.testicle_finetuning import train
    elif args.sft:
        from train.finetuning import train
    else:
        from train.segm_train import train
        
    train(args)
