from collections import defaultdict
import json
from torch.utils.data import Dataset, DataLoader
import os, sys, torch, random
from pathlib import Path
import numpy as np
from torchvision import tv_tensors
from torchvision.transforms.functional import to_pil_image
from utils.transforms import RandomResizedCrop
from utils.utils import (
    crop_ultrasound_pil,
    extract_bbox_ultrasound_cv2,
    organ_to_class_dict,
    dataset_to_organ_dict,
    dataset_for_classification,
    dataset_for_segmentation,
    multi_cls_labels_dict,
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
        data_type="both",
        ccl_crop=False,
        keep_aspect_ratio=True,
        self_norm=False,
        skip_dataset="",
        use_selfaug=True,
    ):
        base_dir = Path(base_dir)
        self.sample_list = []
        self.aug = transforms
        self.out_size = out_size
        self.data_type = data_type
        self.ccl_crop = ccl_crop
        self.keep_aspect_ratio = keep_aspect_ratio
        self.self_norm = self_norm
        self.skip_dataset = skip_dataset
        self.use_selfaug = use_selfaug
        self.base_dir = base_dir
        self.dataset_list = []
        self.sample_by_organ = {k: [] for k in organ_to_class_dict.keys()}
        self.all_bboxes = {}
        self.items = []
        for dataset_dir in base_dir.iterdir():
            if dataset_dir.name in self.dataset_list:
                continue
            elif Path(dataset_dir, split + ".txt").is_file():
                if self.skip_dataset == "":
                    list_path = Path(dataset_dir, f"{split}.txt")
                elif self.skip_dataset in dataset_dir.name and "train" in split:
                    continue
                elif self.skip_dataset in dataset_dir.name and "train" not in split:
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
                    item["multi_cls_label"] = -100
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
                    if dataset_name in dataset_for_classification:
                        item["multi_cls_label"] = multi_cls_labels_dict[dataset_name][
                            int(file_path.parent.name)
                        ]
                    if self.data_type == "both":
                        self.items.append(item)
                    elif (
                        self.data_type == "classification"
                        and dataset_name in dataset_for_classification
                    ):
                        self.items.append(item)
                    elif (
                        self.data_type == "segmentation"
                        and dataset_name in dataset_for_segmentation
                    ):
                        self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        if not self.use_selfaug:
            image = Image.open(item["image_path"]).convert("RGB")
            if self.ccl_crop:
                image, top_left_coords = crop_ultrasound_pil(
                    image, zero_tol=3, min_size_nonblack=200
                )
            image = pil_to_tensor(image)

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
            else:
                mask = pil_to_tensor(Image.open(item["mask_path"]).convert("L"))
                image = resize_pad(
                    image,
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

            if self.skip_dataset != "" and self.skip_dataset in item["image_path"]:
                organ_id = organ_to_class_dict["unknown"]

            image_normed_medsam = (image - image.min()) / torch.clip(
                image.max() - image.min(), min=1e-8, max=None
            )
            if self.self_norm:
                image_normed = self.normalize_tensor_zscore_ignore_black(image)
            else:
                image_normed = v2.functional.normalize(
                    image, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
                )
            sample = {
                "pixel_values": image_normed.to(torch.float),
                "pixel_values_medsam": image_normed_medsam.to(torch.float),
                "organ_id": organ_id,
                "labels": item["multi_cls_label"],
                "masks": mask.to(torch.float).squeeze(),
                "organ_id_metric": organ_id,
            }
            return sample

        else:
            image = pil_to_tensor(Image.open(item["image_path"]).convert("RGB"))
            x1 = image.clone()
            x2 = image.clone()
            if item["mask_path"] is not None:
                mask = pil_to_tensor(Image.open(item["mask_path"]).convert("L"))
            else:
                mask = torch.ones_like(image) * -100

            mask1 = mask.clone()
            mask2 = mask.clone()

            # Resize and pad
            image = resize_pad(image, target_size=512, keep_ratio=True)
            x1 = resize_pad(x1, target_size=512, keep_ratio=True)
            x2 = resize_pad(x2, target_size=512, keep_ratio=True)
            mask1 = resize_pad(mask1, target_size=512, keep_ratio=True)
            mask2 = resize_pad(mask2, target_size=512, keep_ratio=True)

            x1 = v2.RandomHorizontalFlip(0.5)(torch.cat([x1, mask1]))
            mask1 = x1[3:]
            x1 = x1[:3]
            x2 = v2.RandomHorizontalFlip(0.5)(torch.cat([x2, mask2]))
            mask2 = x2[3:]
            x2 = x2[:3]

            w1_flip = torch.Tensor([(x1 == image).all()]).to(torch.bool)
            w2_flip = torch.Tensor([(x2 == image).all()]).to(torch.bool)

            # Random resized crop - create transform instance and apply to both image and mask
            x1, w1_crop = RandomResizedCrop(512, scale=(0.6, 1.2))(to_pil_image(x1))

            mask1_pil = to_pil_image(mask1)
            h, w = mask1_pil.height, mask1_pil.width
            top = int(w1_crop[0] * h)
            left = int(w1_crop[1] * w)
            height = int(w1_crop[2] * h)
            width = int(w1_crop[3] * w)
            mask1 = v2.functional.resized_crop(
                mask1_pil, top, left, height, width, size=(512, 512)
            )

            x2, w2_crop = RandomResizedCrop(512, scale=(0.6, 1.2))(to_pil_image(x2))
            mask2_pil = to_pil_image(mask2)
            h, w = mask2_pil.height, mask2_pil.width
            top = int(w2_crop[0] * h)
            left = int(w2_crop[1] * w)
            height = int(w2_crop[2] * h)
            width = int(w2_crop[3] * w)
            mask2 = v2.functional.resized_crop(
                mask2_pil, top, left, height, width, size=(512, 512)
            )

            # Color jitter - only apply to images, not masks
            b1 = random.uniform(0.6, 1.4)
            b2 = random.uniform(0.6, 1.4)
            c1 = random.uniform(0.6, 1.4)
            c2 = random.uniform(0.6, 1.4)

            x1 = v2.ColorJitter(brightness=[b1, b1], contrast=[c1, c1])(x1)
            x2 = v2.ColorJitter(brightness=[b2, b2], contrast=[c2, c2])(x2)

            b1 = (b1 - 1) / 0.8
            c1 = (c1 - 1) / 0.8

            b2 = (b2 - 1) / 0.8
            c2 = (c2 - 1) / 0.8

            # Calculate differences
            diff = {}
            center1 = w1_crop[:2] + w1_crop[2:] / 2
            center2 = w2_crop[:2] + w2_crop[2:] / 2
            if w1_flip.all():
                center1[1] = 1 - center1[1]
            if w2_flip.all():
                center2[1] = 1 - center2[1]

            diff_crop1 = torch.cat(
                [center1 - center2, w1_crop[2:] - w2_crop[2:]]
            ).unsqueeze(0)
            diff_crop2 = torch.cat(
                [center2 - center1, w2_crop[2:] - w1_crop[2:]]
            ).unsqueeze(0)
            diff["crop"] = torch.cat([diff_crop1, diff_crop2])
            diff_flip1 = (w1_flip == w2_flip).float().unsqueeze(0)
            diff_flip2 = (w1_flip == w2_flip).float().unsqueeze(0)
            diff["flip"] = torch.cat([diff_flip1, diff_flip2])
            diff_jit1 = (torch.Tensor([b1, c1]) - torch.Tensor([b2, c2])).unsqueeze(0)
            diff_jit2 = (torch.Tensor([b2, c2]) - torch.Tensor([b1, c1])).unsqueeze(0)
            diff["jit"] = torch.cat([diff_jit1, diff_jit2])

            mask = torch.stack([pil_to_tensor(mask1), pil_to_tensor(mask2)], dim=0)
            if mask.max() > 1.0:
                mask = mask / 255.0


            organ_id = organ_to_class_dict[item["organ_label"]]

            if self.skip_dataset != "" and self.skip_dataset in item["image_path"]:
                organ_id = organ_to_class_dict["unknown"]

            image_normed_medsam = (image - image.min()) / torch.clip(
                image.max() - image.min(), min=1e-8, max=None
            )
            if self.self_norm:
                x1_normed = self.normalize_tensor_zscore_ignore_black(x1)
                x2_normed = self.normalize_tensor_zscore_ignore_black(x2)
            else:
                x1_normed = v2.functional.normalize(
                    x1, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
                )                
                x2_normed = v2.functional.normalize(
                    x2, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
                )
            image_normed = torch.stack([x1_normed, x2_normed], dim=0)
            sample = {
                "pixel_values": image_normed.to(torch.float),
                "pixel_values_medsam": image_normed_medsam.to(torch.float),
                "organ_id": organ_id,
                "labels": item["multi_cls_label"],
                "masks": mask.to(torch.float).squeeze(),
                "organ_id_metric": organ_id,
                "crop":diff["crop"],
                "flip":diff["flip"],
                "jit":diff["jit"],
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
