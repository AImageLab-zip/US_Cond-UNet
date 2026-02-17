from argparse import Namespace
import hashlib, random, math, time, cv2, torch
import numpy as np
from PIL import Image
from typing import Any, Dict, Optional, Union, Tuple
from torchvision import tv_tensors
from torchvision.transforms import v2
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries
from torch import nn

organ_to_class_dict_for_prediction = {
    "appendix": 0,
    "breast": 1,
    "cardiac": 2,
    "thyroid": 3,
    "fetal": 4,
    "fetal head": 4,
    "fetal_head": 4,
    "kidney": 5,
    "liver": 6,
    "testicles": 7,
    "breast_luminal": 1,
    "breast luminal": 1,
}
organ_to_class_dict = {
    "appendix": 0,
    "breast": 1,
    "cardiac": 2,
    "thyroid": 3,
    "fetal": 4,
    "kidney": 5,
    "liver": 6,
    "testicle": 7,
    "breast_luminal": 1,
    "unknown": -1,
}
class_to_organ_dict = {v: k for k, v in organ_to_class_dict.items()}

dataset_to_organ_dict = {
    "Appendix": "appendix",
    "private_Appendix": "appendix",
    "BUS-BRA": "breast",
    "BUSI": "breast",
    "BUID": "breast",
    "another_Breast": "breast",
    "BUSIS": "breast",
    "private_Breast": "breast",
    "private_Breast_luminal": "breast_luminal",
    "CAMUS": "cardiac",
    "echonet_dataset": "cardiac",
    "private_Cardiac": "cardiac",
    "DDTI": "thyroid",
    "private_Thyroid": "thyroid",
    "Fetal_HC": "fetal",
    "private_Fetal_Head": "fetal",
    "KidneyUS": "kidney",
    "private_Kidney": "kidney",
    "Fatty-Liver": "liver",
    "private_Liver": "liver",
    "private_Testicle": "testicle",
    "public_Testicle_syn": "testicle",
}
dataset_for_classification = [
    "Appendix",
    "BUS-BRA",
    "BUSI",
    "BUSI",
    "BUID",
    "Fatty-Liver",
    "private_Appendix",
    "private_Breast",
    # "private_Breast_luminal",
    "private_Liver",
    "private_Testicle",
]
dataset_for_segmentation = [
    "BUS-BRA",
    "BUSI",
    "BUSIS",
    "BUSIS",
    "another_Breast",
    "BUID",
    "CAMUS",
    "echonet_dataset",
    "DDTI",
    "Fetal_HC",
    "KidneyUS",
    "private_Breast",
    "private_Breast_luminal",
    "private_Cardiac",
    "private_Fetal_Head",
    "private_Kidney",
    "private_Thyroid",
    "private_Testicle",
    "public_Testicle_syn",
    
]

multi_cls_labels_dict = {
    "private_Breast_luminal": [0, 1, 2, 3],
    "private Breast luminal": [0, 1, 2, 3],
    "Breast_luminal": [0, 1, 2, 3],
    "breast_luminal": [0, 1, 2, 3],
    "Breast luminal": [0, 1, 2, 3],
    "BUS-BRA": [4, 5],
    "BUSI": [4, 5],
    "Fatty-Liver": [6, 7],
    "Liver": [6, 7],
    "liver": [6, 7],
    "private_Liver": [6, 7],
    "private Liver": [6, 7],
    "private_Breast": [4, 5],
    "BUID": [4, 5],
    "private Breast": [4, 5],
    "private_Appendix": [8, 9],
    "private Appendix": [8, 9],
    "Appendix": [8, 9],
    "appendix": [8, 9],
    "Breast": [4, 5],
    "breast": [4, 5],
    "private_Testicle": [10, 11],
    "testicle": [10, 11],
}



def resize_pad(
    img: torch.Tensor,
    bbox: Optional[tv_tensors.BoundingBoxes] = None,
    target_size=1024.0,
    keep_ratio=True,
):
    """
    Args
    ----
    img        : Tensor [C, H, W]  or  [1, H, W] for masks
    bbox       : Optional[tv_tensors.BoundingBoxes] with format="XYXY"
    target_size: Target size for the output image
    keep_ratio : If True, maintains aspect ratio and pads. If False, resizes directly to target_size.

    Returns
    -------
    img_padded : Tensor [C, target, target]
    bbox_transformed : Optional[tv_tensors.BoundingBoxes] (if bbox was provided)
    """
    if not keep_ratio:
        # Direct resize to target_size without maintaining aspect ratio
        new_size = [int(target_size), int(target_size)]
        if bbox is not None:
            resized_image, resized_bbox = v2.Resize(size=new_size)(
                tv_tensors.Image(img), bbox
            )
            return resized_image, resized_bbox
        else:
            resized_image = v2.functional.resize(img, size=new_size)
            return resized_image

    # Original logic: maintain aspect ratio with padding
    _, h, w = img.shape

    resize_factor = max(img.shape) / target_size
    new_size = [int(img.shape[-2] / resize_factor), int(img.shape[-1] / resize_factor)]
    if new_size[0] % 2:
        new_size[0] += 1
    if new_size[1] % 2:
        new_size[1] += 1
    if bbox != None:
        resized_image, resized_bbox = v2.Resize(size=new_size)(
            tv_tensors.Image(img), bbox
        )
    else:
        resized_image = v2.functional.resize(img, size=new_size)

    horiz_pad = int((target_size - resized_image.shape[-1]) / 2)
    vertical_pad = int((target_size - resized_image.shape[-2]) / 2)
    if bbox != None:
        padded_image, resized_bbox = v2.Pad(fill=0, padding=[horiz_pad, vertical_pad])(
            resized_image, resized_bbox
        )
    else:
        padded_image = v2.functional.pad(
            resized_image, fill=0, padding=[horiz_pad, vertical_pad]
        )

    if bbox is not None:
        return padded_image, resized_bbox
    else:
        return padded_image


def crop_ultrasound_pil(
    pil_image: Image.Image,
    zero_tol: int = 3,
    connectivity: int = 8,
    min_size_nonblack: int = 64,
) -> Tuple[Optional[Image.Image], Optional[Tuple[int, int]]]:
    """
    Crop ultrasound image using the same logic as the batch processing function.

    Args:
        pil_image: Input PIL Image
        zero_tol: Tolerance for black pixels (0-255)
        connectivity: Connectivity for connected components (4 or 8)
        min_size_nonblack: Minimum size for non-black components
        use_border_touching: Whether to use border-touching heuristic

    Returns:
        Tuple of (cropped_image, (top_left_x, top_left_y))
        Returns (None, None) if no valid crop region found
    """
    # Convert PIL to numpy array
    if pil_image.mode == "RGB":
        rgb_array = np.array(pil_image)
        # Convert RGB to BGR for OpenCV
    else:
        # Convert other modes to RGB first
        rgb_array = np.array(pil_image.convert("RGB"))

    # Get bounding box using the existing function
    bbox = extract_bbox_ultrasound_cv2(
        rgb_array,
        zero_tol=zero_tol,
        connectivity=connectivity,
        min_size_nonblack=min_size_nonblack,
    )

    if bbox is None:
        return pil_image, (0, 0)

    x, y, w, h = bbox

    # Sanity clamp bbox to image bounds
    img_width, img_height = pil_image.size
    x2 = min(x + w, img_width)
    y2 = min(y + h, img_height)
    x = max(0, x)
    y = max(0, y)
    w = max(0, x2 - x)
    h = max(0, y2 - y)

    if w == 0 or h == 0:
        return pil_image, (0, 0)

    # Crop the PIL image
    cropped_pil = pil_image.crop((x, y, x + w, y + h))
    area_pct = 100.0 * (h * w) / (img_width * img_height)
    # print(f"The cropped area is the {area_pct}% of the original size!")
    if area_pct < 10.0:
        # print(f"The cropped area is under 10% no crop was applied!")
        return pil_image, (0, 0)
    return cropped_pil, (x, y)


def extract_bbox_ultrasound_cv2(
    rgb: np.ndarray,
    zero_tol: int = 0,
    connectivity: int = 8,
    min_size_nonblack: int = 64,
):
    """
    Return (x, y, w, h) bbox of largest non-black CC, or None.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Create mask of non-black pixels
    nonblack_mask = (gray > zero_tol).astype(np.uint8) * 255

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        nonblack_mask, connectivity
    )

    if num_labels <= 1:  # Only background
        return None

    # Find largest component (skip label 0 = background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = np.argmax(areas) + 1  # +1 because we skipped label 0

    # Check if it meets minimum size
    if stats[largest_idx, cv2.CC_STAT_AREA] < min_size_nonblack:
        return None

    # Extract bounding box
    x = int(stats[largest_idx, cv2.CC_STAT_LEFT])
    y = int(stats[largest_idx, cv2.CC_STAT_TOP])
    w = int(stats[largest_idx, cv2.CC_STAT_WIDTH])
    h = int(stats[largest_idx, cv2.CC_STAT_HEIGHT])

    return (x, y, w, h)


def generate_run_hash(args: Namespace) -> str:
    """Generate a unique hash for the run based on configuration and timestamp"""
    config_str = f"{args.wandb_run_name}_{args.seed}_{args.learning_rate}_{args.batch_size}_{time.time()}"
    hash_object = hashlib.md5(config_str.encode())
    if args.debug:
        return f"./loggings/debug{hash_object.hexdigest()[:12]}"
    return f"./loggings/{hash_object.hexdigest()[:12]}"


def get_sft_transforms(train: bool, size=512):
    """Get image transforms for training/validation"""
    if train:
        return v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomApply(
                    [v2.RandomResizedCrop(size=size, scale= (0.7, 1.3))],
                    p=0.7,
                ),                
                # v2.RandomApply(
                #     [v2.RandomAffine(degrees=8, shear=10)],
                #     p=0.5,
                # ),
                v2.ColorJitter(brightness=0.4, contrast=0.4),
                v2.ToDtype(torch.float32, scale=False),
            ]
        )
    else:
        return v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=False),
            ]
        )


def compute_dsc(y_true, y_pred):
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)

    intersection = np.sum(y_true * y_pred)
    dsc = 2.0 * intersection / (np.sum(y_true) + np.sum(y_pred) + 1e-6)
    return dsc


def compute_nsd(y_true, y_pred, tolerance=1):
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)

    boundary_true = find_boundaries(y_true, mode="inner")
    boundary_pred = find_boundaries(y_pred, mode="inner")

    distance_true = distance_transform_edt(1 - boundary_true)
    distance_pred = distance_transform_edt(1 - boundary_pred)

    true_in_pred = (boundary_true & (distance_pred <= tolerance)).sum()
    pred_in_true = (boundary_pred & (distance_true <= tolerance)).sum()

    nsd = (true_in_pred + pred_in_true) / (
        boundary_true.sum() + boundary_pred.sum() + 1e-6
    )
    return nsd


def mask_overlap_visualization(pred: torch.Tensor, mask: torch.Tensor):
    """
    Create a color visualization tensor showing overlap between predicted mask and ground truth.
      - Green: correct prediction (TP)
      - Yellow: missed region (FN)
      - Red: false positive (FP)

    Args:
        mask (Tensor): shape (1, H, W) or (H, W), binary ground truth mask
        threshold (float): threshold for logits -> binary mask

    Returns:
        Tensor: RGB visualization tensor of shape (3, H, W), values in [0,1]
    """
    # Ensure proper shape - squeeze all dimensions of size 1
    pred = pred.squeeze()
    mask = mask.squeeze()

    # Ensure they are 2D

    # Logical masks
    tp = (pred == 1) & (mask == 1)  # True positives
    fn = (pred == 0) & (mask == 1)  # False negatives
    fp = (pred == 1) & (mask == 0)  # False positives

    # Create RGB channels
    h, w = pred.shape
    vis = torch.zeros(3, h, w)

    # Assign colors
    vis[0][tp] = 0  # R
    vis[1][tp] = 1  # G
    vis[2][tp] = 0  # B

    vis[0][fn] = 1  # R
    vis[1][fn] = 1  # G
    vis[2][fn] = 0  # B

    vis[0][fp] = 1  # R
    vis[1][fp] = 0  # G
    vis[2][fp] = 0  # B

    return vis


# Code adapted from:
# https://github.com/hankook/AugSelf/

from torch.nn import functional as F


class SSObjective:
    def __init__(self, crop=-1, color=-1, flip=-1, blur=-1, rot=-1, sol=-1, only=False):
        self.only = only
        self.params = [
            ("crop", crop, 4, "regression"),
            ("color", color, 2, "regression"),
            ("flip", flip, 1, "binary_classification"),
            ("blur", blur, 1, "regression"),
            ("rot", rot, 4, "classification"),
            ("sol", sol, 1, "regression"),
        ]

    def __call__(self, ss_predictor, z1, z2, d1, d2, symmetric=True):
        if symmetric:
            z = torch.cat([torch.cat([z1, z2], 1), torch.cat([z2, z1], 1)], 0)
            d = {k: torch.cat([d1[k], d2[k]], 0) for k in d1.keys()}
        else:
            z = torch.cat([z1, z2], 1)
            d = d1

        losses = {"total": 0}
        for name, weight, n_out, loss_type in self.params:
            if weight <= 0:
                continue

            p = ss_predictor[name](z)
            if loss_type == "regression":
                losses[name] = F.mse_loss(torch.tanh(p), d[name])
            elif loss_type == "binary_classification":
                losses[name] = F.binary_cross_entropy_with_logits(p, d[name])
            elif loss_type == "classification":
                losses[name] = F.cross_entropy(p, d[name])
            losses["total"] += losses[name] * weight

        return losses


def load_ss_predictor(n_in, ss_objective, n_hidden=512):
    ss_predictor = {}
    for name, weight, n_out, _ in ss_objective.params:
        if weight > 0:
            ss_predictor[name] = load_mlp(
                n_in * 2, n_hidden, n_out, num_layers=3, last_bn=False
            )

    return nn.ModuleDict(ss_predictor)


def load_mlp(n_in, n_hidden, n_out, num_layers=3, last_bn=True):
    layers = []
    for i in range(num_layers - 1):
        layers.append(nn.Linear(n_in, n_hidden, bias=False))
        layers.append(nn.BatchNorm1d(n_hidden))
        layers.append(nn.ReLU())
        n_in = n_hidden
    layers.append(nn.Linear(n_hidden, n_out, bias=not last_bn))
    if last_bn:
        layers.append(nn.BatchNorm1d(n_out))
    mlp = nn.Sequential(*layers)
    reset_parameters(mlp)
    return mlp


def reset_parameters(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.reset_parameters()

        if isinstance(m, nn.Linear):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.weight, -bound, bound)
            if m.bias is not None:
                nn.init.uniform_(m.bias, -bound, bound)