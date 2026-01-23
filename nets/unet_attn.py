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
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_channels = max_channels

        # Patch embedding with positional encoding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=img_channels,
            embed_dim=emb_dim,
        )

        # Organ embedding
        self.organ_embed = nn.Embedding(n_organs, emb_dim)
        nn.init.normal_(self.organ_embed.weight, mean=0, std=0.02)

        # Layer embedding - NEW
        self.layer_embed = nn.Embedding(n_layers, emb_dim)
        nn.init.normal_(self.layer_embed.weight, mean=0, std=0.02)

        # Learnable influence token
        self.influence_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.trunc_normal_(self.influence_token, std=0.02)

        # Attention with dropout
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim, 
            num_heads=n_heads, 
            dropout=dropout,
            batch_first=True
        )

        # Output projection to MAXIMUM channel size
        self.to_gamma_beta = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 2 * max_channels),
        )

        # Initialize to identity (gamma≈1, beta≈0)
        nn.init.zeros_(self.to_gamma_beta[-1].weight)
        nn.init.constant_(self.to_gamma_beta[-1].bias[:max_channels], 0)  # β
        nn.init.constant_(self.to_gamma_beta[-1].bias[max_channels:], 1)  # γ

    def compute_all_modulations(
        self, 
        original_img: torch.Tensor, 
        organ_id: torch.Tensor,
        layer_configs: list[tuple[int, int]]
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
            if organ_id is not None and (organ_id >= 0).all():
                # Cross-attention mode: organ + layer embedding
                organ_emb = self.organ_embed(organ_id)  # (B, D)
                queries = organ_emb + layer_emb  # (B, D)
            else:
                # Self-attention mode: pooled patches + layer embedding
                # Global average pooling over all patches
                patches_pooled = patches.mean(dim=1)  # (B, D)
                queries = patches_pooled + layer_emb  # (B, D)
            
            queries = queries.unsqueeze(1)  # (B, 1, D)
            
            # Prepend influence token
            influence_tokens = self.influence_token.expand(B, -1, -1)  # (B, 1, D)
            queries = torch.cat([influence_tokens, queries], dim=1)  # (B, 2, D)
            
            # Attention
            attn_out, _ = self.attn(
                query=queries,  # (B, 2, D)
                key=patches,    # (B, N, D)
                value=patches,  # (B, N, D)
            )
            
            # Average over query tokens
            attn_out = attn_out.mean(dim=1)  # (B, D)
            
            # Generate gamma and beta for maximum channels
            gamma_beta = self.to_gamma_beta(attn_out)  # (B, 2 * max_channels)
            beta_max, gamma_max = gamma_beta.chunk(2, dim=-1)  # each (B, max_channels)
            
            # Adaptive pooling to target channel size
            # beta_max = beta_max.unsqueeze(-1)  # (B, max_channels, 1)
            # gamma_max = gamma_max.unsqueeze(-1)  # (B, max_channels, 1)
            
            beta = F.adaptive_avg_pool1d(beta_max, n_channels)  # (B, n_channels, 1)
            gamma = F.adaptive_avg_pool1d(gamma_max, n_channels)  # (B, n_channels, 1)
            
            # Reshape for broadcasting with (B, C, H, W)
            beta = beta.unsqueeze(-1).unsqueeze(-1)   # (B, n_channels, 1, 1)
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

        self.film_layers = nn.ModuleList(
            [FiLMLayer() for _ in out_channels]
        )

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor, gammas: list, betas: list):
        """
        Args:
            x: (B, C, H, W) - features from previous layer
            gammas: list of gamma tensors for each conv block
            betas: list of beta tensors for each conv block
        """
        for conv, film, gamma, beta in zip(self.conv_blocks, self.film_layers, gammas, betas):
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

        self.film_layers = nn.ModuleList(
            [FiLMLayer() for _ in out_channels]
        )

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
        for conv, film, gamma, beta in zip(self.conv_blocks, self.film_layers, gammas, betas):
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
                up_conv = False
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

            if spatial_factor > 1:
                # Need upsampling
                self.distill_upconv = nn.ConvTranspose2d(
                    student_channels,
                    target_channels,
                    kernel_size=3,
                    stride=spatial_factor,
                    padding=1,
                    output_padding=spatial_factor - 1,
                )
            elif spatial_factor == 1:
                # Same spatial size, just adjust channels
                self.distill_upconv = nn.Conv2d(
                    student_channels, target_channels, kernel_size=1
                )
            else:
                # Need downsampling
                downsample_factor = student_spatial // target_spatial
                self.distill_upconv = nn.Sequential(
                    nn.Conv2d(
                        student_channels,
                        target_channels,
                        kernel_size=3,
                        stride=downsample_factor,
                        padding=1,
                    )
                )

            sam_model = sam_model_registry["vit_b"](checkpoint=MEDSAM_BASE_WEIGHTS)
            self.distill_model = MedSAM(
                image_encoder=deepcopy(sam_model.image_encoder),
                mask_decoder=deepcopy(sam_model.mask_decoder),
                prompt_encoder=deepcopy(sam_model.prompt_encoder),
                predict_bboxes=True,
                freeze_image_encoder=0,
            )
            state_dict = load_file(
                "/work/tesi_nmorelli/UUSIC_new/checkpoints/medsam_unfreezed/model.safetensors"
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
        pre_padding = (x.size(-1) % 2**self.depth != 0) or (
            x.size(-2) % 2**self.depth != 0
        ) or (x.size(-3) % 2**self.depth != 0)
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
                self.distill_model.eval()
                up_pixel_values = v2.functional.resize(
                    pixel_values, 1024, v2.InterpolationMode.BICUBIC
                )
                image_embedding = self.distill_model.image_encoder(up_pixel_values)
                image_pe = self.distill_model.prompt_encoder.get_dense_pe()
                # Decode mask
                low_res_masks, _ = self.distill_model.mask_decoder(
                    image_embeddings=image_embedding,  # (B, 256, 64, 64)
                    image_pe=image_pe,  # (1, 256, 64, 64)
                    sparse_prompt_embeddings=self.distill_model.learned_sparse_embeddings,  # (B, 2, 256)
                    dense_prompt_embeddings=self.distill_model.learned_dense_embeddings,  # (B, 256, 64, 64)
                    multimask_output=False,
                )

            up_feat = self.distill_upconv(out_bottleneck)
            distill_loss_emb = self.distill_loss(
                student_logits=up_feat, teacher_logits=image_embedding
            )
            mid_res_masks = v2.functional.resize(
                low_res_masks, 512, v2.InterpolationMode.BICUBIC
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