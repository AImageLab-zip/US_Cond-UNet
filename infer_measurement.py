from tqdm import tqdm
import torch 
from PIL import Image
from torchvision.transforms.v2.functional import pil_to_tensor
from torch.profiler import profile, record_function, ProfilerActivity
from pathlib import Path
from utils.utils import resize_pad,organ_to_class_dict
from utils.paths import *
from nets.segm_net import UNet2DFiLM
from safetensors.torch import load_file


model = UNet2DFiLM(
    in_channels=3,
    num_classes=1,
    n_organs=len(organ_to_class_dict),
    size=32,
    depth=4,
    film_start=0,
)
state_dict = load_file(
    FILMUNET4_CHECKPOINT
)
model.load_state_dict(state_dict)
load_result = model.load_state_dict(state_dict)
print(load_result)

model.cuda()
fake_id = torch.Tensor([0]).long().cuda()
outs = []

paths = [
    str(p)
    for p in Path(
        UUSIC_VAL_DATA
    ).rglob("*png")
]
len(paths)

def normalize_tensor_zscore_ignore_black(tensor: torch.Tensor, epsilon: float = 1e-8):
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



for p in tqdm(paths):
    image = Image.open(p).convert("RGB")
    image = pil_to_tensor(image).cuda()
    image = resize_pad(
        image,
        target_size=512,
        keep_ratio=True,
    )
    image = normalize_tensor_zscore_ignore_black(image.float()).unsqueeze(0)
    # break

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for p in tqdm(paths):
        image = Image.open(p).convert("RGB")
        image = pil_to_tensor(image).cuda()
        image = resize_pad(
            image,
            target_size=512,
            keep_ratio=True,
        )
        image = normalize_tensor_zscore_ignore_black(image.float()).unsqueeze(0)
        with torch.no_grad():
            outs.append(model(image, fake_id))

print(prof.key_averages().table(sort_by="cuda_time_total"))
            