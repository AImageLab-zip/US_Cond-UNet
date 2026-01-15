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
    dataset_for_classification,
    dataset_for_segmentation,
    multi_cls_labels_dict,
    resize_pad,
)
from PIL import Image
from torchvision.transforms.v2.functional import pil_to_tensor, center_crop
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from tqdm import tqdm 
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle

class ImageDataset(Dataset):
    def __init__(self, items, processor):
        self.items = items
        self.processor = processor
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        image = load_image(item['image_path'])
        inputs = self.processor(images=image, return_tensors="pt")
        # Remove batch dimension added by processor
        return {k: v.squeeze(0) for k, v in inputs.items()}, idx

def collate_fn(batch):
    """Custom collate to handle variable-sized inputs"""
    inputs_list, indices = zip(*batch)
    # Stack inputs
    batched_inputs = {
        key: torch.stack([inp[key] for inp in inputs_list])
        for key in inputs_list[0].keys()
    }
    return batched_inputs, list(indices)

class USdatasetOmni(Dataset):
    def __init__(
        self,
        base_dir,
        split,
        transforms=None,
        out_size=1024.0,
        data_type="both",
        ccl_crop=False,
        keep_aspect_ratio=True,
        self_norm=False,
        include_testicles=False,
        testicle_split="",
        self_id=False,
        num_clusters=10,
        kmeans_model=None,  # NEW: Pass trained KMeans model from training dataset
    ):
        base_dir = Path(base_dir)
        self.sample_list = []
        self.aug = transforms
        self.out_size = out_size
        self.data_type = data_type
        self.ccl_crop = ccl_crop
        self.keep_aspect_ratio = keep_aspect_ratio
        self.self_norm = self_norm
        self.self_id = self_id
        self.num_clusters = num_clusters
        self.kmeans_model = kmeans_model  # NEW: Store the passed model
        self.dataset_list = []
        self.sample_by_organ = {k: [] for k in organ_to_class_dict.keys()}
        self.all_bboxes = {}
        self.mean = (
            torch.Tensor([123.675, 116.28, 103.53])
            .view(3, 1, 1)
            .expand(3, out_size, out_size)
        )
        self.std = (
            torch.Tensor([58.395, 57.12, 57.375])
            .view(3, 1, 1)
            .expand(3, out_size, out_size)
        )
        self.items = []
        self.label_dict = {
            # 0: appendix
            0: ["Appendix", "appendix", "Vermiform appendix", "vermiform appendix"],
            # 1: breast
            1: ["Breast", "breast", "Mammary gland", "mammary gland"],
            # 2: cardiac (related to the heart)
            2: ["Cardiac", "cardiac", "Heart", "heart"],
            # 3: thyroid
            3: ["Thyroid", "thyroid", "Thyroid gland", "thyroid gland"],
            # 4: fetal (related to the fetus)
            4: ["Fetal", "fetal", "Fetus", "fetus"],
            # 5: kidney
            5: ["Kidney", "kidney", "Renal", "renal"],
            # 6: liver
            6: ["Liver", "liver", "Hepatic", "hepatic"],
            # 7: testicles
            7: [
                "Testicles",
                "testicles",
                "Testicle",
                "testicle",
                "Testis",
                "testis",
                "Testes",
            ],
            # 8: breast_luminal
            8: [
                "Breast Luminal",
                "breast luminal",
                "Luminal Breast",
                "luminal breast",
                "breast_luminal",
            ],
        }
        for dataset_dir in base_dir.iterdir():
            if dataset_dir.name in self.dataset_list:
                continue
            elif (
                Path(dataset_dir, split + ".txt").is_file()
                or Path(dataset_dir, split + f"{testicle_split}.txt").is_file()
            ):
                if "Testicle" in dataset_dir.name and not include_testicles:
                    continue
                elif "Testicle" in dataset_dir.name and include_testicles:
                    list_path = Path(dataset_dir, split + f"{testicle_split}.txt")
                else:
                    list_path = Path(dataset_dir, split + ".txt")
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

        if self.self_id:
            self.__init_self_ids__()

    def __init_self_ids__(self, batch_size=32, num_workers=16):
        pretrained_model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        model_fe = AutoModel.from_pretrained(
            pretrained_model_name, 
            device_map="auto", 
        )
        model_fe.eval()
        
        # Create dataset and dataloader
        dataset = ImageDataset(self.items, processor)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=6,
            collate_fn=collate_fn,
            prefetch_factor=6,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print("Computing dino embeddings")
        with torch.inference_mode():
            for inputs, indices in tqdm(dataloader, total=len(dataloader)):
                inputs = {k: v.to(model_fe.device) for k, v in inputs.items()}
                outputs = model_fe(**inputs)
                
                pooled_output = outputs.pooler_output.cpu()
                
                # Assign embeddings back to items
                for idx, embedding in zip(indices, pooled_output):
                    self.items[idx]['self_id'] = embedding

        embeddings = np.stack([item['self_id'].numpy() for item in self.items])
        
        # NEW: Check if we have a pre-trained model to use
        if self.kmeans_model is not None:
            print(f"Using pre-trained KMeans model with {self.num_clusters} clusters")
            cluster_labels = self.kmeans_model.predict(embeddings)
        else:
            print(f"Training new KMeans model with {self.num_clusters} clusters")
            self.kmeans_model = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
            cluster_labels = self.kmeans_model.fit_predict(embeddings)
            
        for i, item in enumerate(self.items):
            item['cluster_id'] = cluster_labels[i]
            item['cluster_center_distance'] = np.linalg.norm(
                embeddings[i] - self.kmeans_model.cluster_centers_[cluster_labels[i]]
            )
    
    def get_kmeans_model(self):
        """
        Returns the trained KMeans model.
        Use this to pass the model from training dataset to test dataset.
        
        Returns:
            KMeans model or None if not initialized
        """
        return self.kmeans_model
                
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

        if self.self_norm:
            image = (image * self.std) + self.mean
            image = self.normalize_tensor_zscore_ignore_black(image)

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
        label_id = organ_to_class_dict[item["organ_label"]]
        return {
            "pixel_values": image.to(torch.float),
            "organ_id": item['cluster_id'].item() if self.self_id else label_id,
            "labels": item["multi_cls_label"],
            "masks": mask.to(torch.float).squeeze(),
            "bbox_coords": unormalized_bbox_coords,
            "organ_id_metric": label_id,
        }

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