import torch, wandb
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from nets.segm_net import (
    ConvBlock,
    DownConvBlock,
    UpConvBlock,
    DiceBCELoss,
    pad_to_2d,
    unpad_2d,
)
from segment_anything import sam_model_registry
from copy import deepcopy
from nets.segm_net import UNet2DFiLM, MedSAM, MedSAMPrompt, DistillationLoss
from safetensors.torch import load_file
from utils.paths import *
from torchvision.transforms import v2
import ptwt  # pytorch_wavelets
import torch
import torch.nn as nn
from einops import rearrange


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

import ptwt
import torch
import torch.nn as nn
from einops import rearrange

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
        wavelet='haar',
        mode='zero',
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
        n_patches_per_band = patches_per_dim ** 2  # 256
        
        # Total patches: selected subbands × n_patches_per_band
        self.n_patches = len(self.bands) * n_patches_per_band
        self.n_patches_per_band = n_patches_per_band
        
        # Separate projections for each subband
        self.proj_LL = nn.Conv2d(in_channels, embed_dim, 
                                 kernel_size=patch_size, stride=patch_size)
        self.proj_LH = nn.Conv2d(in_channels, embed_dim,
                                 kernel_size=patch_size, stride=patch_size)
        self.proj_HL = nn.Conv2d(in_channels, embed_dim,
                                 kernel_size=patch_size, stride=patch_size)
        self.proj_HH = nn.Conv2d(in_channels, embed_dim,
                                 kernel_size=patch_size, stride=patch_size)
        
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
            band_patches = rearrange(band_patches, 'b d h w -> b (h w) d')
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
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.max_channels = max_channels

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

        # Layer embedding - NEW
        self.layer_embed = nn.Embedding(n_layers, emb_dim)
        nn.init.normal_(self.layer_embed.weight, mean=0, std=0.02)

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
            queries = q[:, None, :]  # [B, 1, D] for attention

            # Prepend influence token
            influence_tokens = self.influence_token.expand(B, -1, -1)  # (B, 1, D)
            queries = torch.cat([influence_tokens, queries], dim=1)  # (B, 2, D)

            # Attention
            attn_out, _ = self.attn(
                query=queries,  # (B, 2, D)
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

        return modulations


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


class UNet2DAttn(nn.Module):
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
        attn_start: int = 0,
        use_attn: bool = True,
        img_size: int = 256,
        patch_size: int = 16,
        emb_dim: int = 256,
        n_heads: int = 8,
        distill: bool = False,
        use_dwt: bool = True,
        wavelet: str = "haar",
        dwt_bands: list[str] | None = None,
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
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = num_classes
        self.size = size
        self.depth = depth
        self.attn_start = max(0, int(attn_start))
        self.use_attn = use_attn
        self.n_organs = n_organs
        self.img_size = img_size
        self.distill = distill

        self.criterion = DiceBCELoss()

        # Compute max channels needed
        max_channels = size * (2 ** (depth + 1))

        # Create shared attention modulator
        if self.use_attn:
            # Count total number of layers that will use attention
            n_attn_layers = 0
            for i in range(depth):
                if i >= self.attn_start:
                    n_attn_layers += 2  # encoder has 2 conv blocks per level
            n_attn_layers += 2  # bottleneck
            for i in range(depth):
                if i >= self.attn_start:
                    n_attn_layers += 2  # decoder has 2 conv blocks per level

            self.shared_attn = SharedAttnModulator(
                n_organs=n_organs,
                n_layers=n_attn_layers,
                max_channels=max_channels,
                img_size=img_size,
                patch_size=patch_size,
                emb_dim=emb_dim,
                n_heads=n_heads,
                use_dwt=use_dwt,
                wavelet=wavelet,
                dwt_bands=dwt_bands,
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

        if self.distill:
            student_channels = 2048 // (2 ** (5 - self.depth))
            student_spatial = 32 * (2 ** (5 - self.depth))

            # Target: 256 channels, 64x64 (teacher output)
            target_channels = 256
            target_spatial = 64

            # Calculate upsampling factor
            spatial_factor = target_spatial // student_spatial

            self.distill_adapter = nn.Conv2d(student_channels, target_channels, kernel_size=1)

            sam_model = sam_model_registry["vit_b"](checkpoint=MEDSAM_BASE_WEIGHTS)
            self.distill_model = MedSAM(
                image_encoder=deepcopy(sam_model.image_encoder),
                mask_decoder=deepcopy(sam_model.mask_decoder),
                prompt_encoder=deepcopy(sam_model.prompt_encoder),
                predict_bboxes=True,
                freeze_image_encoder=0,
            )
            state_dict = load_file(
                "/work/phd_ultrasounds/UUSIC_new/checkpoints/medsam_unfreezed/model.safetensors"
            )
            self.distill_model.load_state_dict(state_dict)
            load_result = self.distill_model.load_state_dict(state_dict)
            for p in self.distill_model.parameters():
                p.requires_grad = False
            self.distill_model.eval()
            print(f"Loaded MedSam teacher model and loaded weights:\n{load_result}")
            self.distill_loss = DistillationLoss()

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

    def forward(
        self,
        pixel_values,
        organ_id=None,
        labels=None,
        masks=None,
        bbox_coords=None,
        organ_id_metric=None,
        teacher_embedding = None,
        teacher_mask = None,
        **kwargs,
    ):
        """
        Forward pass through the network.

        Args:
            pixel_values: Input images (B, C, H, W) - MUST be 256x256
            organ_id: Organ type IDs for attention conditioning (B,)
            masks: Ground truth masks (B, H, W)
        """
        x = pixel_values
        original_img = pixel_values

        # Pre-compute all gamma/beta if using attention
        if self.use_attn:
            layer_configs = self._build_layer_configs()
            modulations = self.shared_attn.compute_all_modulations(
                original_img, organ_id, layer_configs
            )

            # Convert to list for easy indexing
            mod_list = [modulations[i] for i in range(len(layer_configs))]
            mod_idx = 0

        feat_list = []

        # Padding if needed
        pre_padding = (
            (x.size(-1) % 2**self.depth != 0)
            or (x.size(-2) % 2**self.depth != 0)
            or (x.size(-3) % 2**self.depth != 0)
        )
        if pre_padding:
            x, pads = pad_to_2d(x, 2**self.depth)
            original_img, _ = pad_to_2d(original_img, 2**self.depth)

        # Encoder
        if isinstance(self.encoder["0"], DownConvBlockFiLM):
            gammas = [mod_list[mod_idx][0], mod_list[mod_idx + 1][0]]
            betas = [mod_list[mod_idx][1], mod_list[mod_idx + 1][1]]
            mod_idx += 2
            out, feat = self.encoder["0"](x, gammas, betas)
        else:
            out, feat = self.encoder["0"](x)
        feat_list.append(feat)

        for key in list(self.encoder.keys())[1:]:
            if isinstance(self.encoder[key], DownConvBlockFiLM):
                gammas = [mod_list[mod_idx][0], mod_list[mod_idx + 1][0]]
                betas = [mod_list[mod_idx][1], mod_list[mod_idx + 1][1]]
                mod_idx += 2
                out, feat = self.encoder[key](out, gammas, betas)
            else:
                out, feat = self.encoder[key](out)
            feat_list.append(feat)

        # Bottleneck
        if isinstance(self.bottleneck, UpConvBlockFiLM):
            gammas = [mod_list[mod_idx][0], mod_list[mod_idx + 1][0]]
            betas = [mod_list[mod_idx][1], mod_list[mod_idx + 1][1]]
            mod_idx += 2
            out = self.bottleneck(out, gammas, betas)
        else:
            out = self.bottleneck(out)
        out_bottleneck = out

        # Decoder
        for key in self.decoder:
            concat_feat = torch.cat((out, feat_list[int(key)]), dim=1)

            if isinstance(self.decoder[key], UpConvBlockFiLM):
                gammas = [mod_list[mod_idx][0], mod_list[mod_idx + 1][0]]
                betas = [mod_list[mod_idx][1], mod_list[mod_idx + 1][1]]
                mod_idx += 2
                out = self.decoder[key](concat_feat, gammas, betas)
            else:
                out = self.decoder[key](concat_feat)

            del feat_list[int(key)]

        # Output
        out = self.out_layer(out)

        if pre_padding:
            out = unpad_2d(out, pads).squeeze(1)

        # Calculate loss if masks provided
        if masks is not None:
            loss = self.criterion(out, masks)
        else:
            loss = 0.0

        if self.distill:
            with torch.no_grad():
                up_pixel_values = v2.functional.resize(
                    pixel_values, 1024, v2.InterpolationMode.BICUBIC
                )
                image_embedding = self.distill_model.image_encoder(up_pixel_values)
                image_pe = self.distill_model.prompt_encoder.get_dense_pe()
                low_res_masks, _ = self.distill_model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=self.distill_model.learned_sparse_embeddings,
                    dense_prompt_embeddings=self.distill_model.learned_dense_embeddings,
                    multimask_output=False,
                )
                mid_res_masks = v2.functional.resize(
                    low_res_masks,
                    out.shape[-1],
                    v2.InterpolationMode.BICUBIC,
                )
            student_resized = nn.functional.interpolate(
                out_bottleneck,
                size=(image_embedding.shape[-1], image_embedding.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            student_resized = nn.functional.interpolate(
                out_bottleneck,
                size=(image_embedding.shape[-1], image_embedding.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            up_feat = self.distill_adapter(student_resized)
            distill_loss_emb = self.distill_loss(
                student_logits=up_feat, teacher_logits=image_embedding
            )
            distill_loss_logits = self.distill_loss(
                student_logits=out, teacher_logits=mid_res_masks.squeeze(1)
            )

            # print(f"distill_loss: {distill_loss}")
            if wandb.run is not None:
                wandb.log(
                    {
                        "distill_loss_emb": distill_loss_emb["loss"].item(),
                        "distill_loss_logits": distill_loss_logits["loss"].item(),
                    },
                    commit=False,
                )

            loss = loss + distill_loss_emb["loss"]

        return {
            "loss": loss,
            "logits": out,
            "labels": masks,
            "organ_id": organ_id,
            "organ_id_metric": organ_id_metric,
        }
