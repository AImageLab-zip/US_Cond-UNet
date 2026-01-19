import numpy as np
import random, torch
from torch.utils.data import Subset
from data_classes.datasets import USdatasetOmni, organ_to_class_dict
from torchvision.transforms import v2
from sklearn.model_selection import StratifiedShuffleSplit
from utils.utils import get_sft_transforms
import copy


def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_strata(dataset):
    """
    Returns:
      strata: np.ndarray of shape (N,) with combined labels "organId|multiLabel"
      organs: np.ndarray of shape (N,) with organ ids alone (for fallback)
    """
    organs = []
    multi = []
    for it in dataset.items:  # use metadata; no image I/O
        organ_id = organ_to_class_dict[it["organ_label"]]
        organs.append(int(organ_id))
        multi.append(int(it["multi_cls_label"]))  # may be -100 for 'not available'
    organs = np.asarray(organs, dtype=int)
    multi = np.asarray(multi, dtype=int)
    strata = np.array([f"{o}|{m}" for o, m in zip(organs, multi)])
    return strata, organs

def stratified_80_20_indices(dataset, seed: int = 42):
    strata, organs = make_strata(dataset)

    # First try: strict stratification on (organ_id, multi_cls_label)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    idx = np.arange(len(dataset))
    try:
        train_idx, val_idx = next(sss.split(idx, strata))
        return train_idx.tolist(), val_idx.tolist()
    except ValueError as e:
        # Common cause: at least one stratum has < 2 samples
        print(f"[split] Falling back to organ-only stratification because: {e}")
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, val_idx = next(sss2.split(idx, organs))
        return train_idx.tolist(), val_idx.tolist()

def build_train_val_datasets(
    DATA_DIR,
    args,
    seed: int = 42,
    get_sft_transforms_ = None,
    id_file_name = 'train_cls'
):
    set_all_seeds(seed)

    # 1) Build the full "train" split as you already do
    train_full = USdatasetOmni(
        DATA_DIR,
        id_file_name,
        transforms=get_sft_transforms(train=True) if get_sft_transforms_ is None else get_sft_transforms_(train=True),
        data_type=args.dataset_type,
        out_size=args.dataset_size,
        ccl_crop=args.use_ccl_crop,
        keep_aspect_ratio=args.keep_aspect_ratio,
        include_testicles=False,
        self_id = args.self_id,
        use_cluster_id = args.use_cluster_id,
        enc_type = args.enc_type,
        num_clusters = args.num_clusters,
        id_dropout=args.id_dropout
    )

    # 2) Get reproducible stratified indices (80/20)
    tr_idx, va_idx = stratified_80_20_indices(train_full, seed=seed)

    # 3) Create Subset datasets (keep same transforms; you can add different aug for val if needed)
    train_dataset = Subset(train_full, tr_idx)

    # If you want different transforms for validation, rebuild a second base dataset w/ val transforms:
    # val_base = USdatasetOmni(
    #     DATA_DIR,
    #     id_file_name,  # same source list, we'll index it with va_idx
    #     transforms=get_sft_transforms(train=False) if get_sft_transforms_ is None else get_sft_transforms_(train=False),
    #     data_type=args.dataset_type,
    #     out_size=args.dataset_size,
    #     ccl_crop=args.use_ccl_crop,
    #     keep_aspect_ratio=args.keep_aspect_ratio,
    #     include_testicles=True,
    #     self_id = args.self_id,
    #     num_clusters = args.num_clusters,
    # )
    val_base = copy.deepcopy(train_full)
    val_base.aug = get_sft_transforms(train=False) if get_sft_transforms_ is None else get_sft_transforms_(train=False)
    val_base.id_dropout = 0.0
    val_dataset = Subset(val_base, va_idx)

    print(f"[split] Total: {len(train_full)} | Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    return train_dataset, val_dataset


