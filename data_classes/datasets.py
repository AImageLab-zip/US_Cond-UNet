from collections import defaultdict
import json
from torch.utils.data import Dataset, DataLoader
import os, sys, torch, random
from pathlib import Path
import numpy as np
from torchvision import tv_tensors
from utils.utils import (
    crop_ultrasound_pil,
    extract_bbox_ultrasound_cv2,
    organ_to_class_dict,
    dataset_to_organ_dict,
    dataset_for_segmentation,
    resize_pad,
)
from PIL import Image
from torchvision.transforms.v2.functional import pil_to_tensor, center_crop
from transformers import AutoImageProcessor, AutoModel, CLIPVisionModel
from transformers.image_utils import load_image
from tqdm import tqdm 
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
from typing import Literal, Optional, Union
from torchvision.transforms import v2
from functools import partial

class USdatasetOmni(Dataset):
    def __init__(
        self,
        base_dir,
        split,
        transforms=None,
        out_size=1024,
        ccl_crop=False,
        keep_aspect_ratio=True,
        self_norm=False,
        skip_dataset = "",
        id_dropout: float = 0.0,
    ):
        base_dir = Path(base_dir)
        self.sample_list = []
        self.aug = transforms
        self.out_size = out_size
        self.ccl_crop = ccl_crop
        self.keep_aspect_ratio = keep_aspect_ratio
        self.self_norm = self_norm
        self.skip_dataset = skip_dataset
        self.id_dropout = id_dropout  
        self.base_dir = base_dir
        self.dataset_list = []
        self.sample_by_organ = {k: [] for k in organ_to_class_dict.keys()}
        self.all_bboxes = {}
        self.items = []
        for dataset_dir in base_dir.iterdir():
            if dataset_dir.name in self.dataset_list:
                continue
            elif (
                Path(dataset_dir, split + ".txt").is_file()
            ):  
                if self.skip_dataset == "":
                    list_path = Path(dataset_dir, f"{split}.txt")
                elif self.skip_dataset in dataset_dir.name and 'train' in split:
                    continue
                elif self.skip_dataset in dataset_dir.name and 'train' not in split:
                    list_path = Path(dataset_dir, f"{split}.txt")
                else:
                    list_path = Path(dataset_dir, f"{split}.txt")
                
                self.dataset_list.append(dataset_dir.name)
                with open(list_path, "r") as f:

                    files = [
                        os.path.join(dataset_dir.name, line.strip().split("/")[-1])
                        for line in f.readlines()
                    ]

                    self.sample_list.extend(files)
                    self.sample_by_organ[
                        dataset_to_organ_dict[dataset_dir.name]
                    ].extend(files)
                if Path(base_dir, dataset_dir.name, "bboxes.json").is_file():
                    with open(
                        Path(base_dir, dataset_dir.name, "bboxes.json"), "r"
                    ) as f:
                        self.all_bboxes[dataset_dir.name] = json.load(f)
            else:
                print(
                    f"Warning, {dataset_dir.name} was found without a {split}.txt file"
                )

        to_search_dirs = []
        for dataset_name in self.dataset_list:
            for subdir in Path(base_dir, dataset_name).iterdir():
                if subdir.is_dir() and "mask" not in str(subdir):
                    to_search_dirs.append(subdir)

        for subdir in to_search_dirs:
            for file_path in subdir.rglob("*"):
                dataset_name = file_path.parent.parent.name
                if f"{dataset_name}/{file_path.name}" in self.sample_list:
                    item = {}
                    item["image_path"] = str(file_path)
                    item["mask_path"] = None
                    item["bbox_regr"] = [-100, -100, -100, -100]
                    item["organ_label"] = dataset_to_organ_dict[dataset_name]

                    if dataset_name in dataset_for_segmentation:
                        if Path(
                            file_path.parent.parent, "masks", file_path.name
                        ).is_file():
                            item["mask_path"] = (
                                f"{file_path.parent.parent}/masks/{file_path.name}"
                            )
                            item["bbox_regr"] = self.all_bboxes[dataset_name][
                                f"segmentation/{dataset_name}/masks/{file_path.name}"
                            ]["bbox_prompt"]

                        else:
                            print(
                                f"Warning: {file_path.parent.parent}/masks/{file_path.name} not found"
                            )
                    if dataset_name in dataset_for_segmentation:
                        self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        if self.ccl_crop:
            image, top_left_coords = crop_ultrasound_pil(
                image, zero_tol=3, min_size_nonblack=200
            )
        image = pil_to_tensor(image)

        bbox_coords = torch.tensor([item["bbox_regr"]], dtype=torch.float32)

        # Handle mask
        if item["mask_path"] is None:
            mask = (
                torch.ones(
                    (1, int(self.out_size), int(self.out_size)), dtype=torch.float
                )
                * 255
            )
            image = resize_pad(
                image, target_size=self.out_size, keep_ratio=self.keep_aspect_ratio
            )
            unormalized_bbox_coords = None
        else:
            mask = pil_to_tensor(Image.open(item["mask_path"]).convert("L"))
            bbox_t = tv_tensors.BoundingBoxes(
                bbox_coords,
                format="XYXY",
                canvas_size=image.shape[-2:],
            )
            image, unormalized_bbox_coords = resize_pad(
                image,
                bbox=bbox_t,
                target_size=self.out_size,
                keep_ratio=self.keep_aspect_ratio,
            )

        mask = resize_pad(
            mask, target_size=self.out_size, keep_ratio=self.keep_aspect_ratio
        )

        # Apply augmentations
        if self.aug is not None:
            image = tv_tensors.Image(image)
            mask = tv_tensors.Mask(mask)

            if unormalized_bbox_coords != None:
                image, mask, unormalized_bbox_coords = self.aug(
                    image, mask, unormalized_bbox_coords
                )
            else:
                image, mask = self.aug(image, mask)

            if mask.max() > 1.0:
                mask = mask / 255.0


        if len(mask.shape) < 3:
            mask = mask.unsqueeze(0)

        if item["mask_path"] is None:
            mask = (
                torch.ones(
                    (1, int(self.out_size), int(self.out_size)), dtype=torch.float
                )
                * -100
            )
            unormalized_bbox_coords = torch.tensor(
                [-100, -100, -100, -100], dtype=torch.float32
            ).unsqueeze(0)
        
        organ_id = organ_to_class_dict[item["organ_label"]]
        if self.id_dropout != 0.0 and random.random() < self.id_dropout:
            organ_id = organ_to_class_dict['unknown']
        
        if self.skip_dataset != "" and self.skip_dataset in item["image_path"]:
            organ_id = organ_to_class_dict['unknown']
        
        image_normed_medsam = (image - image.min())/torch.clip(image.max() - image.min(), min=1e-8, max=None)
        if self.self_norm:
            image_normed = self.normalize_tensor_zscore_ignore_black(image)
        else:
            image_normed = v2.functional.normalize(image, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
        sample = {
            "pixel_values": image_normed.to(torch.float),
            "pixel_values_medsam": image_normed_medsam.to(torch.float),
            "organ_id": organ_id,
            "masks": mask.to(torch.float).squeeze(),
            "bbox_coords": unormalized_bbox_coords,
            "organ_id_metric": organ_id,
        }
        return sample

    def normalize_tensor_zscore_ignore_black(
        self, tensor: torch.Tensor, epsilon: float = 1e-8
    ):
        """
        Z-score normalize a tensor, ignoring black pixels.

        Returns:
            Normalized tensor with mean≈0, std≈1 for non-black pixels
        """
        if tensor.dim() == 2:
            mask = tensor > 0
        elif tensor.dim() == 3:
            mask = (
                (tensor > 0).any(dim=0)
                if tensor.shape[0] in [1, 3]
                else (tensor > 0).any(dim=-1)
            )

        if mask.any():
            valid_pixels = tensor[mask] if tensor.dim() == 2 else tensor[:, mask]
            mean_val = valid_pixels.mean()
            std_val = valid_pixels.std()

            if std_val > epsilon:
                normalized = (tensor - mean_val) / std_val
            else:
                normalized = tensor - mean_val
        else:
            normalized = tensor.clone()

        return normalized
