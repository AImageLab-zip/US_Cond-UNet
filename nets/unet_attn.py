import torch, wandb
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from nets.segm_net import (
    ConvBlock,
    DownConvBlock,
    UpConvBlock,
    DiceBCELoss,
)
import ptwt
import numpy as np
from nets.unet_base import BaseUnet


def canonicalize_mask_batch(gt_masks: torch.Tensor, canon_res: int = 32, pad: int = 4):
    """
    Args:
        gt_masks: (B, H, W) binary float tensor (0/1)
        canon_res: output resolution (int)
        pad: pixels of padding around bbox (optional)
    Returns:
        canon_masks: (B, canon_res, canon_res) float tensor in [0,1]
    """
    B, H, W = gt_masks.shape
    device = gt_masks.device
    canon_masks = torch.zeros(
        (B, canon_res, canon_res), device=device, dtype=gt_masks.dtype
    )

    for i in range(B):
        m = gt_masks[i]
        nz = torch.nonzero(m, as_tuple=False)
        if nz.numel() == 0:
            # empty mask -> keep zeros
            continue
        y_min = int(nz[:, 0].min().clamp(0, H - 1).item())
        y_max = int(nz[:, 0].max().clamp(0, H - 1).item())
        x_min = int(nz[:, 1].min().clamp(0, W - 1).item())
        x_max = int(nz[:, 1].max().clamp(0, W - 1).item())

        # pad bbox
        y0 = max(0, y_min - pad)
        y1 = min(H, y_max + pad + 1)
        x0 = max(0, x_min - pad)
        x1 = min(W, x_max + pad + 1)

        crop = m[y0:y1, x0:x1].unsqueeze(0).unsqueeze(0)  # (1,1,hc,wc)
        # Resize to canonical resolution
        crop_resized = F.interpolate(
            crop, size=(canon_res, canon_res), mode="bilinear", align_corners=False
        )
        canon_masks[i] = crop_resized[0, 0]

    return canon_masks  # (B, canon_res, canon_res)


def canonicalize_mask_batch_normalized(
    gt_masks: torch.Tensor,
    canon_res: int = 32,
    target_scale: float = 0.7,  # Target mask to fill 70% of canonical space
):
    """
    Canonical masks with consistent scale normalization.
    """
    B, H, W = gt_masks.shape
    device = gt_masks.device
    canon_masks = torch.zeros(
        (B, canon_res, canon_res), device=device, dtype=gt_masks.dtype
    )

    for i in range(B):
        m = gt_masks[i]
        nz = torch.nonzero(m, as_tuple=False)
        if nz.numel() == 0:
            continue

        y_coords = nz[:, 0].float()
        x_coords = nz[:, 1].float()

        # Center of mass
        com_y = y_coords.mean()
        com_x = x_coords.mean()

        # Bbox dimensions
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()
        bbox_h = (y_max - y_min + 1).item()
        bbox_w = (x_max - x_min + 1).item()

        # Compute crop size to achieve target scale
        max_dim = max(bbox_h, bbox_w)
        crop_size = int(
            max_dim / target_scale
        )  # Scale up to make mask fill target_scale

        # Create square crop centered on COM
        half_crop = crop_size // 2
        y0 = int(com_y.item()) - half_crop
        y1 = y0 + crop_size
        x0 = int(com_x.item()) - half_crop
        x1 = x0 + crop_size

        # Clamp and pad
        y0_clamped = max(0, y0)
        y1_clamped = min(H, y1)
        x0_clamped = max(0, x0)
        x1_clamped = min(W, x1)

        crop = m[y0_clamped:y1_clamped, x0_clamped:x1_clamped]

        # Padding
        pad_top = y0_clamped - y0
        pad_bottom = crop_size - (y1_clamped - y0_clamped) - pad_top
        pad_left = x0_clamped - x0
        pad_right = crop_size - (x1_clamped - x0_clamped) - pad_left

        crop = F.pad(
            crop.unsqueeze(0).unsqueeze(0),
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0,
        )

        # Resize
        crop_resized = F.interpolate(
            crop, size=(canon_res, canon_res), mode="bilinear", align_corners=False
        )
        canon_masks[i] = crop_resized[0, 0]

    return canon_masks


class PatchEmbed(nn.Module):
    """Convert image to patches and embed them with positional encoding"""

    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Positional encoding - CRITICAL for spatial understanding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, N, D) where N = num_patches
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = rearrange(x, "b c h w -> b (h w) c")  # (B, N, D)
        x = x + self.pos_embed  # Add positional encoding
        return x


class DWTPatchEmbed(nn.Module):
    """
    Apply DWT and create patch embeddings from subbands.
    Interface matches PatchEmbed to minimize changes to SharedAttnModulator.

    DWT decomposes (B, C, H, W) into:
    - LL: Low-freq approximation (B, C, H/2, W/2)
    - LH, HL, HH: High-freq details (B, C, H/2, W/2 each)
    """

    def __init__(
        self,
        img_size=256,
        patch_size=8,
        in_channels=3,
        embed_dim=256,
        wavelet="haar",
        mode="zero",
        bands: list[str] | None = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.wavelet = wavelet
        self.mode = mode
        self.available_bands = ["LL", "LH", "HL", "HH"]
        if not bands:
            self.bands = list(self.available_bands)
        else:
            normalized = []
            seen = set()
            for band in bands:
                band = band.upper()
                if band in seen:
                    continue
                if band not in self.available_bands:
                    raise ValueError(
                        f"Invalid DWT band '{band}'. Valid bands: {self.available_bands}"
                    )
                normalized.append(band)
                seen.add(band)
            self.bands = normalized

        # After 1-level DWT, each subband is img_size/2
        dwt_size = img_size // 2  # 128 for img_size=256

        # Number of patches per subband
        patches_per_dim = dwt_size // patch_size  # 128/8 = 16
        n_patches_per_band = patches_per_dim**2  # 256

        # Total patches: selected subbands × n_patches_per_band
        self.n_patches = len(self.bands) * n_patches_per_band
        self.n_patches_per_band = n_patches_per_band

        # Separate projections for each subband
        self.proj_LL = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.proj_LH = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.proj_HL = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.proj_HH = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Shared positional encoding for all patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Learnable subband type embeddings
        self.subband_type_embed = nn.Parameter(torch.zeros(4, embed_dim))
        nn.init.trunc_normal_(self.subband_type_embed, std=0.02)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - input image
        Returns:
            patches: (B, N, D) where N = len(bands) * (H/2/patch_size)^2
        """
        B, C, H, W = x.shape

        # Apply DWT using ptwt - correct API
        # wavedec2 returns coefficients in format: [LL, (LH, HL, HH)]
        coeffs = ptwt.wavedec2(x, wavelet=self.wavelet, mode=self.mode, level=1)

        # Extract subbands
        # coeffs[0] is LL (approximation)
        # coeffs[1] is tuple of (LH, HL, HH) detail coefficients
        LL = coeffs[0]  # (B, C, H/2, W/2)
        LH, HL, HH = coeffs[1]  # Each (B, C, H/2, W/2)

        subband_tensors = {
            "LL": (LL, self.proj_LL, 0),
            "LH": (LH, self.proj_LH, 1),
            "HL": (HL, self.proj_HL, 2),
            "HH": (HH, self.proj_HH, 3),
        }
        patches_list = []
        for band in self.bands:
            band_tensor, proj, band_idx = subband_tensors[band]
            band_patches = proj(band_tensor)  # (B, embed_dim, H/2/P, W/2/P)
            band_patches = rearrange(band_patches, "b d h w -> b (h w) d")
            band_patches = band_patches + self.subband_type_embed[band_idx]
            patches_list.append(band_patches)

        # Concatenate selected subbands
        patches = torch.cat(patches_list, dim=1)

        # Add positional encoding
        patches = patches + self.pos_embed

        return patches


class FiLMLayer(nn.Module):
    """Simple FiLM layer that applies gamma * x + beta modulation"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
        """
        Args:
            x: (B, C, H, W) - input features
            gamma: (B, C, 1, 1) - scaling factors
            beta: (B, C, 1, 1) - bias terms
        Returns:
            (B, C, H, W) - modulated features
        """
        return gamma * x + beta


class SharedAttnModulator(nn.Module):
    """
    Shared attention modulator that computes gamma/beta for all layers at once.
    Uses layer_id embeddings to differentiate between layers.
    """

    def __init__(
        self,
        n_organs: int,
        n_layers: int,
        max_channels: int,
        img_size: int = 256,
        patch_size: int = 16,
        img_channels: int = 3,
        emb_dim: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_dwt: bool = True,
        wavelet: str = "haar",
        dwt_bands: list[str] | None = None,
        use_shape: bool = False,
        shape_res=32,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.max_channels = max_channels
        self.use_shape = use_shape
        self.shape_res = shape_res

        if use_dwt:
            self.patch_embed = DWTPatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=img_channels,
                embed_dim=emb_dim,
                wavelet=wavelet,
                bands=dwt_bands,
            )
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=img_channels,
                embed_dim=emb_dim,
            )

        # Organ embedding
        self.organ_embed = nn.Embedding(
            n_organs + 1, emb_dim
        )  # the +1 is for the unknown organ
        nn.init.normal_(self.organ_embed.weight, mean=0, std=0.02)

        # Layer embedding
        self.layer_embed = nn.Embedding(n_layers, emb_dim)
        nn.init.normal_(self.layer_embed.weight, mean=0, std=0.02)

        if self.use_shape:
            # Shape embedding
            self.shape_embed = nn.Embedding(
                n_organs + 1, emb_dim
            )  # the +1 is for the unknown organ
            nn.init.normal_(self.shape_embed.weight, mean=0, std=0.02)
            self.shape_proj = nn.Linear(self.emb_dim, self.shape_res**2)

        # Learnable influence token
        self.influence_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.trunc_normal_(self.influence_token, std=0.02)

        # Attention with dropout
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        # Shared components for all layers
        self.shared_norm = nn.LayerNorm(emb_dim)

        # Per-layer linear projections
        self.layer_linears = nn.ModuleList(
            [nn.Linear(emb_dim, emb_dim) for _ in range(n_layers)]
        )

        # Shared components after layer-specific linear
        self.shared_gelu = nn.GELU()
        self.shared_dropout = nn.Dropout(dropout)
        self.shared_output = nn.Linear(emb_dim, 2 * max_channels)

        # Initialize to identity (gamma≈1, beta≈0)
        nn.init.zeros_(self.shared_output.weight)
        nn.init.constant_(self.shared_output.bias[:max_channels], 0)  # β
        nn.init.constant_(self.shared_output.bias[max_channels:], 1)  # γ

    def compute_all_modulations(
        self,
        original_img: torch.Tensor,
        organ_id: torch.Tensor,
        layer_configs: list[tuple[int, int]],
    ):
        """
        Compute gamma and beta for all layers at once.

        Args:
            original_img: (B, 3, 256, 256) - original input image
            organ_id: (B,) - organ type IDs (if >= 0, use organ embedding)
            layer_configs: list of (layer_id, n_channels) tuples

        Returns:
            dict: {layer_id: (gamma, beta)} where gamma/beta are (B, C, 1, 1)
        """
        B = original_img.shape[0]

        # Get patch embeddings from original image (shared across all layers)
        patches = self.patch_embed(original_img)  # (B, N, D)

        modulations = {}
        projected_shapes = []
        for layer_id, n_channels in layer_configs:
            # Get layer embedding
            layer_emb = self.layer_embed(
                torch.tensor([layer_id], device=original_img.device)
            )  # (1, D)
            layer_emb = layer_emb.expand(B, -1)  # (B, D)

            # Determine query based on organ_id validity
            mask = organ_id >= 0  # [B]
            q_org = self.organ_embed(
                organ_id.clamp(min=0)
            )  # [B, D] (dummy for unknown)
            q_img = self.organ_embed(
                torch.tensor(
                    [self.organ_embed.weight.shape[0] - 1], device=original_img.device
                ).expand(B)
            )
            q = torch.where(mask[:, None], q_org, q_img) + layer_emb  # [B, D]
            queries_organ = q[:, None, :]  # [B, 1, D] for attention

            influence_tokens = self.influence_token.expand(B, -1, -1)  # (B, 1, D)

            if self.use_shape:
                q_org = self.shape_embed(
                    organ_id.clamp(min=0)
                )  # [B, D] (dummy for unknown)
                q_img = self.shape_embed(
                    torch.tensor(
                        [self.shape_embed.weight.shape[0] - 1],
                        device=original_img.device,
                    ).expand(B)
                )
                q = torch.where(mask[:, None], q_org, q_img)  # [B, D]
                queries_shape = q[:, None, :]  # [B, 1, D] for attention
                queries_shape_detached = queries_shape.detach()
                flattened_shape = self.shape_proj(
                    queries_shape[:, 0, :]
                )  # Use non-detached for shape loss
                projected_shape = flattened_shape.reshape(
                    B, self.shape_res, self.shape_res
                )
                projected_shapes.append(projected_shape)

                # Prepend influence token
                queries = torch.cat(
                    [influence_tokens, queries_organ, queries_shape_detached], dim=1
                )  # (B, 3, D)

            else:
                queries = torch.cat(
                    [influence_tokens, queries_organ], dim=1
                )  # (B, 3, D)

            # Attention
            attn_out, _ = self.attn(
                query=queries,  # (B, 3, D)
                key=patches,  # (B, N, D)
                value=patches,  # (B, N, D)
            )

            # Average over query tokens
            attn_out = attn_out.mean(dim=1)  # (B, D)

            # Generate gamma and beta using layer-specific linear
            x = self.shared_norm(attn_out)
            x = self.layer_linears[layer_id](x)  # Layer-specific transformation
            x = self.shared_gelu(x)
            x = self.shared_dropout(x)
            gamma_beta = self.shared_output(x)  # (B, 2 * max_channels)

            beta_max, gamma_max = gamma_beta.chunk(2, dim=-1)  # each (B, max_channels)

            # Adaptive pooling to target channel size
            beta = F.adaptive_avg_pool1d(beta_max, n_channels)  # (B, n_channels)
            gamma = F.adaptive_avg_pool1d(gamma_max, n_channels)  # (B, n_channels)

            # Reshape for broadcasting with (B, C, H, W)
            beta = beta.unsqueeze(-1).unsqueeze(-1)  # (B, n_channels, 1, 1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, n_channels, 1, 1)

            modulations[layer_id] = (gamma, beta)

        if projected_shapes == []:
            projected_shapes = torch.Tensor([])
        else:
            projected_shapes = torch.stack(projected_shapes)

        return modulations, projected_shapes.to(original_img.device)


class DownConvBlockFiLM(nn.Module):
    """
    Encoder block with FiLM modulation.
    Conv → FiLM → Conv → FiLM → Pool
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: list[int],
        conv_kwargs={"kernel_size": 3, "stride": 1, "padding": 1},
    ):
        super().__init__()
        assert len(in_channels) == len(out_channels)

        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(in_ch, out_ch, conv_kwargs)
                for in_ch, out_ch in zip(in_channels, out_channels)
            ]
        )

        self.film_layers = nn.ModuleList([FiLMLayer() for _ in out_channels])

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor, gammas: list, betas: list):
        """
        Args:
            x: (B, C, H, W) - features from previous layer
            gammas: list of gamma tensors for each conv block
            betas: list of beta tensors for each conv block
        """
        for conv, film, gamma, beta in zip(
            self.conv_blocks, self.film_layers, gammas, betas
        ):
            x = conv(x)
            x = film(x, gamma, beta)
        return self.pool(x), x


class UpConvBlockFiLM(nn.Module):
    """
    Decoder block with FiLM modulation.
    Conv → FiLM → Conv → FiLM → (optional) ConvTranspose2d
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: list[int],
        up_conv: bool = True,
        conv_kwargs: dict = {"kernel_size": 3, "stride": 1, "padding": 1},
        upconv_kwargs: dict = {"kernel_size": 2, "stride": 2},
    ):
        super().__init__()
        assert len(in_channels) == len(out_channels)

        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(in_ch, out_ch, conv_kwargs)
                for in_ch, out_ch in zip(in_channels, out_channels)
            ]
        )

        self.film_layers = nn.ModuleList([FiLMLayer() for _ in out_channels])

        self.up_conv = up_conv
        if self.up_conv:
            self.up_conv_op = nn.ConvTranspose2d(
                out_channels[-1], out_channels[-1], **upconv_kwargs
            )

    def forward(self, x: torch.Tensor, gammas: list, betas: list):
        """
        Args:
            x: (B, C, H, W)
            gammas: list of gamma tensors for each conv block
            betas: list of beta tensors for each conv block
        """
        for conv, film, gamma, beta in zip(
            self.conv_blocks, self.film_layers, gammas, betas
        ):
            x = conv(x)
            x = film(x, gamma, beta)

        if self.up_conv:
            x = self.up_conv_op(x)

        return x


class UNet2DAttn(BaseUnet):
    """
    UNet with shared attention-based conditioning.
    Computes all gamma/beta at the start of forward pass.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        n_organs: int,
        size: int = 32,
        depth: int = 3,
        *,
        attn_start: int = 0,
        use_attn: bool = True,
        img_size: int = 512,
        patch_size: int = 8,
        emb_dim: int = 768,
        n_heads: int = 8,
        distill: bool = False,
        distill_unet: bool = False,
        medsam_teacher_ckpt: str = "/work/phd_ultrasounds/UUSIC_new/checkpoints/medsam_unfreezed/model.safetensors",
        unet_teacher_ckpt: str = "/work/phd_ultrasounds/UUSIC_new/checkpoints/unet5_attn/model.safetensors",
        unet_teacher_kwargs: dict | None = None,
        use_dwt: bool = True,
        wavelet: str = "haar",
        dwt_bands: list[str] | None = None,
        use_shape: bool = False,
        shape_res: int = 64,
    ):
        """
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            n_organs: Number of organ types
            size: Base number of channels
            depth: Number of encoder/decoder levels
            attn_start: Level where attention starts (0-based)
            use_attn: If False, use plain Conv blocks
            img_size: Size of input images (assumed square)
            patch_size: Patch size for attention
            emb_dim: Embedding dimension for attention
            n_heads: Number of attention heads
        """
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            n_organs=n_organs,
            size=size,
            depth=depth,
            attn_start=attn_start,
            use_attn=use_attn,
            img_size=img_size,
            patch_size=patch_size,
            emb_dim=emb_dim,
            n_heads=n_heads,
            distill=distill,
            distill_unet=distill_unet,
            medsam_teacher_ckpt=medsam_teacher_ckpt,
            unet_teacher_ckpt=unet_teacher_ckpt,
            unet_teacher_kwargs=unet_teacher_kwargs,
            use_dwt=use_dwt,
            wavelet=wavelet,
            dwt_bands=dwt_bands,
            use_shape=use_shape,
            shape_res=shape_res,
        )

    def _build_model(
        self,
        *,
        attn_start: int = 0,
        use_attn: bool = True,
        img_size: int = 512,
        patch_size: int = 8,
        emb_dim: int = 768,
        n_heads: int = 8,
        distill: bool = False,
        distill_unet: bool = False,
        medsam_teacher_ckpt: str = "/work/phd_ultrasounds/UUSIC_new/checkpoints/medsam_unfreezed/model.safetensors",
        unet_teacher_ckpt: str = "/work/phd_ultrasounds/UUSIC_new/checkpoints/unet5_attn/model.safetensors",
        unet_teacher_kwargs: dict | None = None,
        use_dwt: bool = True,
        wavelet: str = "haar",
        dwt_bands: list[str] | None = None,
        use_shape: bool = False,
        shape_res: int = 64,
        **kwargs,
    ):
        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected UNet2DAttn kwargs: {unknown}")

        self.attn_start = max(0, int(attn_start))
        self.use_attn = bool(use_attn)
        self.use_shape = bool(use_shape)
        self.img_size = int(img_size)
        self.shape_res = int(shape_res)
        self.criterion = DiceBCELoss()
        self.steps_counter = 0

        # Compute max channels needed
        max_channels = self.size * (2 ** (self.depth + 1))

        # Create shared attention modulator
        if self.use_attn:
            # Count total number of layers that will use attention
            n_attn_layers = 0
            for i in range(self.depth):
                if i >= self.attn_start:
                    n_attn_layers += 2  # encoder has 2 conv blocks per level
            n_attn_layers += 2  # bottleneck
            for i in range(self.depth):
                if i >= self.attn_start:
                    n_attn_layers += 2  # decoder has 2 conv blocks per level

            self.shared_attn = SharedAttnModulator(
                n_organs=self.n_organs,
                n_layers=n_attn_layers,
                max_channels=max_channels,
                img_size=self.img_size,
                patch_size=patch_size,
                emb_dim=emb_dim,
                n_heads=n_heads,
                use_dwt=use_dwt,
                wavelet=wavelet,
                dwt_bands=dwt_bands,
                use_shape=use_shape,
                shape_res=shape_res,
            )

        # ---------------- Encoder ----------------
        self.encoder = nn.ModuleDict()

        # First encoder block
        if self.use_attn and 0 >= self.attn_start:
            self.encoder["0"] = DownConvBlockFiLM(
                [self.in_channels, self.size],
                [self.size, self.size * 2],
            )
        else:
            self.encoder["0"] = DownConvBlock(
                [self.in_channels, self.size], [self.size, self.size * 2]
            )

        # Remaining encoder blocks
        for i in range(1, self.depth):
            in_ch = [self.size * (2**i), self.size * (2**i)]
            out_ch = [self.size * (2**i), self.size * (2 ** (i + 1))]
            key = str(i)

            if self.use_attn and i >= self.attn_start:
                self.encoder[key] = DownConvBlockFiLM(in_ch, out_ch)
            else:
                self.encoder[key] = DownConvBlock(in_ch, out_ch)

        # ---------------- Bottleneck ----------------
        if self.use_attn:
            self.bottleneck = UpConvBlockFiLM(
                [self.size * (2**self.depth), self.size * (2**self.depth)],
                [self.size * (2**self.depth), self.size * (2 ** (self.depth + 1))],
            )
        else:
            self.bottleneck = UpConvBlock(
                [self.size * (2**self.depth), self.size * (2**self.depth)],
                [self.size * (2**self.depth), self.size * (2 ** (self.depth + 1))],
            )

        # ---------------- Decoder ----------------
        self.decoder = nn.ModuleDict()

        for i in range(self.depth, 1, -1):
            use_attn_at_level = self.use_attn and (i - 1) >= self.attn_start

            if use_attn_at_level:
                self.decoder[str(i - 1)] = UpConvBlockFiLM(
                    [
                        self.size * (2 ** (i + 1)) + self.size * (2**i),
                        self.size * (2**i),
                    ],
                    [self.size * (2**i), self.size * (2**i)],
                )
            else:
                self.decoder[str(i - 1)] = UpConvBlock(
                    [
                        self.size * (2 ** (i + 1)) + self.size * (2**i),
                        self.size * (2**i),
                    ],
                    [self.size * (2**i), self.size * (2**i)],
                )

        # Final decoder block
        if self.use_attn and 0 >= self.attn_start:
            self.decoder["0"] = UpConvBlockFiLM(
                [self.size * 4 + self.size * 2, self.size * 2],
                [self.size * 2, self.size * 2],
                up_conv=False,
            )
        else:
            self.decoder["0"] = UpConvBlock(
                [self.size * 4 + self.size * 2, self.size * 2],
                [self.size * 2, self.size * 2],
                up_conv=False,
            )

        self.out_layer = ConvBlock(
            self.size * 2,
            self.out_channels,
            conv_kwargs={"kernel_size": 1, "stride": 1, "padding": 0},
        )
        self._init_distillation(
            distill=distill,
            distill_unet=distill_unet,
            medsam_teacher_ckpt=medsam_teacher_ckpt,
            unet_teacher_ckpt=unet_teacher_ckpt,
            unet_teacher_kwargs=unet_teacher_kwargs,
        )

    def _build_layer_configs(self):
        """
        Build list of (layer_id, n_channels) for all layers that use attention.
        Layer IDs are sequential: 0, 1, 2, ...
        """
        configs = []
        layer_id = 0

        # Encoder layers
        for i in range(self.depth):
            if self.use_attn and i >= self.attn_start:
                # Each encoder block has 2 conv outputs
                configs.append((layer_id, self.size * (2**i)))
                layer_id += 1
                configs.append((layer_id, self.size * (2 ** (i + 1))))
                layer_id += 1

        # Bottleneck (2 conv blocks)
        if self.use_attn:
            configs.append((layer_id, self.size * (2**self.depth)))
            layer_id += 1
            configs.append((layer_id, self.size * (2 ** (self.depth + 1))))
            layer_id += 1

        # Decoder layers
        for i in range(self.depth - 1, -1, -1):
            if self.use_attn and i >= self.attn_start:
                # Each decoder block has 2 conv outputs
                configs.append((layer_id, self.size * (2 ** (i + 1))))
                layer_id += 1
                configs.append((layer_id, self.size * (2 ** (i + 1))))
                layer_id += 1

        return configs

    def _prepare_forward(
        self,
        *,
        pixel_values: torch.Tensor,
        organ_id: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        forward_ctx = {
            "mod_list": None,
            "mod_idx": 0,
            "projected_shapes": None,
        }
        if self.use_attn:
            layer_configs = self._build_layer_configs()
            modulations, projected_shapes = self.shared_attn.compute_all_modulations(
                pixel_values, organ_id, layer_configs
            )
            forward_ctx["mod_list"] = [modulations[i] for i in range(len(layer_configs))]
            forward_ctx["projected_shapes"] = projected_shapes
        return forward_ctx

    def _next_modulation(self, forward_ctx: dict):
        mod_idx = forward_ctx["mod_idx"]
        mod_list = forward_ctx["mod_list"]
        gammas = [mod_list[mod_idx][0], mod_list[mod_idx + 1][0]]
        betas = [mod_list[mod_idx][1], mod_list[mod_idx + 1][1]]
        forward_ctx["mod_idx"] = mod_idx + 2
        return gammas, betas

    def _encode(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        organ_id: torch.Tensor | None = None,
        forward_ctx: dict | None = None,
    ):
        if isinstance(layer, DownConvBlockFiLM):
            gammas, betas = self._next_modulation(forward_ctx)
            return layer(x, gammas, betas)
        return layer(x)

    def _bottleneck(
        self,
        x: torch.Tensor,
        organ_id: torch.Tensor | None = None,
        forward_ctx: dict | None = None,
    ):
        if isinstance(self.bottleneck, UpConvBlockFiLM):
            gammas, betas = self._next_modulation(forward_ctx)
            return self.bottleneck(x, gammas, betas)
        return self.bottleneck(x)

    def _decode(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        organ_id: torch.Tensor | None = None,
        forward_ctx: dict | None = None,
    ):
        if isinstance(layer, UpConvBlockFiLM):
            gammas, betas = self._next_modulation(forward_ctx)
            return layer(x, gammas, betas)
        return layer(x)

    def _apply_auxiliary_losses(
        self,
        *,
        loss: torch.Tensor | float,
        logits: torch.Tensor,
        masks: torch.Tensor | None,
        organ_id: torch.Tensor | None = None,
        forward_ctx: dict | None = None,
        **kwargs,
    ):
        self.steps_counter += 1
        if not self.use_shape or masks is None:
            return loss

        projected_shapes = forward_ctx.get("projected_shapes", None)
        if projected_shapes is None or projected_shapes.numel() == 0:
            return loss

        reduced_masks = canonicalize_mask_batch_normalized(
            masks, canon_res=self.shape_res
        )
        loss_shape = torch.tensor([0.0], requires_grad=True).to(logits.device)
        for projected_shape in projected_shapes:
            loss_shape = loss_shape + self.criterion(projected_shape, reduced_masks)

        loss_shape = loss_shape / projected_shapes.shape[0]
        if wandb.run is not None:
            wandb.log({"shape_loss": loss_shape.item()}, commit=False)
        loss = loss + loss_shape

        if self.steps_counter % 200 == 0:
            with torch.no_grad():
                proj_tokens = torch.sigmoid(
                    self.shared_attn.shape_proj(
                        self.shared_attn.shape_embed.weight[:8].clone().detach()
                    )
                    .reshape(8, self.shape_res, self.shape_res)
                    .detach()
                )
            proj_tokens = (proj_tokens > 0.5).to(torch.float32)
            proj_vis = (
                proj_tokens.view(2, 4, self.shape_res, self.shape_res)
                .permute(0, 2, 1, 3)
                .reshape(self.shape_res * 2, self.shape_res * 4)
            )
            wandb.log({"projected_tokens": wandb.Image(proj_vis.unsqueeze(0))})
            print("log")

        return loss
