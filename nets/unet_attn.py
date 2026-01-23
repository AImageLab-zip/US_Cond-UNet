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
        
        # Positional encoding
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


class GlobalTransformerModulator(nn.Module):
    """
    Single global transformer that modulates all UNet layers.
    Keys/Values: patches from original image
    Queries: combination of layer features + layer embeddings
    """

    def __init__(
        self,
        n_organs: int,
        layer_configs: list,  # List of (layer_id, feature_channels) tuples
        img_size: int = 256,
        patch_size: int = 16,
        img_channels: int = 3,
        emb_dim: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
        n_transformer_layers: int = 4,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.layer_configs = layer_configs
        self.n_layers = len(layer_configs)

        # Patch embedding (computed once, shared across all layers)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=img_channels,
            embed_dim=emb_dim,
        )

        # Organ embedding
        self.organ_embed = nn.Embedding(n_organs, emb_dim)
        nn.init.normal_(self.organ_embed.weight, mean=0, std=0.02)

        # Layer embeddings - each UNet layer gets a unique embedding
        self.layer_embed = nn.Embedding(self.n_layers, emb_dim)
        nn.init.normal_(self.layer_embed.weight, mean=0, std=0.02)
        
        # Pre-register layer ID tensors as buffers (not parameters, won't be trained)
        for layer_id, _ in layer_configs:
            self.register_buffer(
                f'layer_id_{layer_id}',
                torch.tensor([layer_id], dtype=torch.long)
            )

        # Feature projections for each layer (spatial features -> embedding space)
        self.feat_to_emb = nn.ModuleDict()
        for layer_id, feat_channels in layer_configs:
            self.feat_to_emb[str(layer_id)] = nn.Sequential(
                nn.Conv2d(feat_channels, emb_dim, kernel_size=1),
                nn.GroupNorm(num_groups=min(32, emb_dim), num_channels=emb_dim),
                nn.GELU(),
            )

        # Single global transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers,
        )

        # Output projections for each layer (embedding -> gamma/beta)
        self.to_gamma_beta = nn.ModuleDict()
        for layer_id, feat_channels in layer_configs:
            proj = nn.Sequential(
                nn.LayerNorm(emb_dim),
                nn.Linear(emb_dim, emb_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(emb_dim, 2 * feat_channels),
            )
            # Initialize to identity transformation
            nn.init.zeros_(proj[-1].weight)
            nn.init.constant_(proj[-1].bias[:feat_channels], 0)  # β
            nn.init.constant_(proj[-1].bias[feat_channels:], 1)  # γ
            self.to_gamma_beta[str(layer_id)] = proj

        # Cache for image patches (computed once per forward pass)
        self.cached_patches = None
        self.cached_batch_size = None



    def forward(
        self,
        layer_id: int,
        features: torch.Tensor,
        original_img: torch.Tensor,
        organ_id: torch.Tensor = None,
    ):
        """
        Args:
            layer_id: Which UNet layer is being modulated (0, 1, 2, ...)
            features: (B, C, H, W) - features from the UNet layer
            original_img: (B, 3, 256, 256) - original input image
            organ_id: (B,) - organ type IDs (optional)
        
        Returns:
            Modulated features (B, C, H, W)
        """
        B, C, H, W = features.shape

        # Get or compute cached image patches (Keys & Values)
        patches = self.patch_embed(original_img)  # (B, N_patches, D)

        # Build query from: features + layer embedding + (optional) organ embedding
        # 1. Pool and project features
        feat_pooled = F.adaptive_avg_pool2d(features, (1, 1))  # (B, C, 1, 1)
        feat_emb = self.feat_to_emb[str(layer_id)](feat_pooled)  # (B, D, 1, 1)
        feat_emb = feat_emb.flatten(2).transpose(1, 2)  # (B, 1, D)

        # 2. Add layer embedding (use pre-registered buffer)
        layer_id_tensor = getattr(self, f'layer_id_{layer_id}')
        layer_emb = self.layer_embed(layer_id_tensor).unsqueeze(0).expand(B, -1, -1)  # (B, 1, D)

        # 3. Optionally add organ embedding
        if organ_id is not None and (organ_id >= 0).all():
            organ_emb = self.organ_embed(organ_id).unsqueeze(1)  # (B, 1, D)
            query = feat_emb + layer_emb + organ_emb  # (B, 1, D)
        else:
            query = feat_emb + layer_emb  # (B, 1, D)

        # Concatenate query with patches for transformer
        # The transformer will process: [query_token, patch_1, ..., patch_N]
        transformer_input = torch.cat([query, patches], dim=1)  # (B, 1+N, D)

        # Global transformer processing
        transformer_output = self.transformer(transformer_input)  # (B, 1+N, D)

        # Extract the query token output (first token)
        query_output = transformer_output[:, 0, :]  # (B, D)

        # Generate gamma and beta for this layer
        gamma_beta = self.to_gamma_beta[str(layer_id)](query_output)  # (B, 2C)
        beta, gamma = gamma_beta.chunk(2, dim=-1)  # each (B, C)

        # Reshape for broadcasting
        beta = beta.unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        # Apply modulation
        return gamma * features + beta



class DownConvBlockAttn(nn.Module):
    """Encoder block with global transformer modulation"""

    def __init__(
        self,
        in_channels: list[int],
        out_channels: list[int],
        layer_ids: list[int],  # List of layer IDs, one per conv block
        global_transformer: GlobalTransformerModulator,
        conv_kwargs={"kernel_size": 3, "stride": 1, "padding": 1},
    ):
        super().__init__()
        assert len(in_channels) == len(out_channels) == len(layer_ids)
        self.layer_ids = layer_ids
        # self.global_transformer = global_transformer
        self.__dict__["global_transformer"] = global_transformer
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(in_ch, out_ch, conv_kwargs)
                for in_ch, out_ch in zip(in_channels, out_channels)
            ]
        )

        self.pool = nn.MaxPool2d(2, 2)

    def forward(
        self, x: torch.Tensor, original_img: torch.Tensor, organ_id: torch.Tensor
    ):
        for i, conv in enumerate(self.conv_blocks):
            x = conv(x)
            # Apply global transformer modulation after each conv
            x = self.global_transformer(self.layer_ids[i], x, original_img, organ_id)
        
        return self.pool(x), x


class UpConvBlockAttn(nn.Module):
    """Decoder block with global transformer modulation"""

    def __init__(
        self,
        in_channels: list[int],
        out_channels: list[int],
        layer_ids: list[int],  # List of layer IDs, one per conv block
        global_transformer: GlobalTransformerModulator,
        up_conv: bool = True,
        conv_kwargs: dict = {"kernel_size": 3, "stride": 1, "padding": 1},
        upconv_kwargs: dict = {"kernel_size": 2, "stride": 2},
    ):
        super().__init__()
        assert len(in_channels) == len(out_channels) == len(layer_ids)
        self.layer_ids = layer_ids
        # self.global_transformer = global_transformer
        self.__dict__["global_transformer"] = global_transformer

        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(in_ch, out_ch, conv_kwargs)
                for in_ch, out_ch in zip(in_channels, out_channels)
            ]
        )

        self.up_conv = up_conv
        if self.up_conv:
            self.up_conv_op = nn.ConvTranspose2d(
                out_channels[-1], out_channels[-1], **upconv_kwargs
            )

    def forward(
        self, x: torch.Tensor, original_img: torch.Tensor, organ_id: torch.Tensor
    ):
        for i, conv in enumerate(self.conv_blocks):
            x = conv(x)
            # Apply global transformer modulation after each conv
            x = self.global_transformer(self.layer_ids[i], x, original_img, organ_id)

        if self.up_conv:
            x = self.up_conv_op(x)

        return x


class UNet2DAttn(nn.Module):
    """
    UNet with a single global transformer that modulates all layers.
    The transformer uses image patches as keys/values and layer-specific
    queries that incorporate layer embeddings.
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
        n_transformer_layers: int = 4,
        distill: bool = False,
    ):
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

        # Build layer configuration for global transformer
        layer_configs = []
        layer_id = 0
        
        # Encoder layers
        layer_configs.extend([(layer_id, size), (layer_id + 1, size * 2)])
        layer_id += 2
        
        for i in range(1, depth):
            layer_configs.extend([
                (layer_id, size * (2**i)),
                (layer_id + 1, size * (2**(i+1)))
            ])
            layer_id += 2
        
        # Bottleneck
        layer_configs.extend([
            (layer_id, size * (2**depth)),
            (layer_id + 1, size * (2**(depth+1)))
        ])
        layer_id += 2
        
        # Decoder layers
        for i in range(depth, 0, -1):
            layer_configs.extend([
                (layer_id, size * (2**i)),
                (layer_id + 1, size * (2**i))
            ])
            layer_id += 2

        # Create single global transformer
        if self.use_attn:
            self.global_transformer = GlobalTransformerModulator(
                n_organs=n_organs,
                layer_configs=layer_configs,
                img_size=img_size,
                patch_size=patch_size,
                emb_dim=emb_dim,
                n_heads=n_heads,
                n_transformer_layers=n_transformer_layers,
            )
        else:
            self.global_transformer = None

        # Track current layer ID
        self.current_layer_id = 0

        # Encoder
        self.encoder = nn.ModuleDict()
        
        if self.use_attn and 0 >= self.attn_start:
            self.encoder["0"] = DownConvBlockAttn(
                [self.in_channels, self.size],
                [self.size, self.size * 2],
                layer_ids=[0, 1],
                global_transformer=self.global_transformer,
            )
        else:
            self.encoder["0"] = DownConvBlock(
                [self.in_channels, self.size], [self.size, self.size * 2]
            )

        layer_id = 2
        for i in range(1, self.depth):
            in_ch = [self.size * (2**i), self.size * (2**i)]
            out_ch = [self.size * (2**i), self.size * (2 ** (i + 1))]
            key = str(i)

            if self.use_attn and i >= self.attn_start:
                self.encoder[key] = DownConvBlockAttn(
                    in_ch,
                    out_ch,
                    layer_ids=[layer_id, layer_id + 1],
                    global_transformer=self.global_transformer,
                )
            else:
                self.encoder[key] = DownConvBlock(in_ch, out_ch)
            layer_id += 2

        # Bottleneck
        if self.use_attn:
            self.bottleneck = UpConvBlockAttn(
                [self.size * (2**self.depth), self.size * (2**self.depth)],
                [self.size * (2**self.depth), self.size * (2 ** (self.depth + 1))],
                layer_ids=[layer_id, layer_id + 1],
                global_transformer=self.global_transformer,
            )
        else:
            self.bottleneck = UpConvBlock(
                [self.size * (2**self.depth), self.size * (2**self.depth)],
                [self.size * (2**self.depth), self.size * (2 ** (self.depth + 1))],
            )
        layer_id += 2

        # Decoder
        self.decoder = nn.ModuleDict()

        for i in range(self.depth, 1, -1):
            use_attn_at_level = self.use_attn and (i - 1) >= self.attn_start

            if use_attn_at_level:
                self.decoder[str(i - 1)] = UpConvBlockAttn(
                    [self.size * (2 ** (i + 1)) + self.size * (2**i), self.size * (2**i)],
                    [self.size * (2**i), self.size * (2**i)],
                    layer_ids=[layer_id, layer_id + 1],
                    global_transformer=self.global_transformer,
                )
            else:
                self.decoder[str(i - 1)] = UpConvBlock(
                    [self.size * (2 ** (i + 1)) + self.size * (2**i), self.size * (2**i)],
                    [self.size * (2**i), self.size * (2**i)],
                )
            layer_id += 2

        # Final decoder
        if self.use_attn and 0 >= self.attn_start:
            self.decoder["0"] = UpConvBlockAttn(
                [self.size * 4 + self.size * 2, self.size * 2],
                [self.size * 2, self.size * 2],
                layer_ids=[layer_id, layer_id + 1],
                global_transformer=self.global_transformer,
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

    def _enc_forward(self, layer, x, original_img, organ_id):
        if isinstance(layer, DownConvBlockAttn):
            return layer(x, original_img, organ_id)
        else:
            return layer(x)

    def _dec_forward(self, layer, x, original_img, organ_id):
        if isinstance(layer, UpConvBlockAttn):
            return layer(x, original_img, organ_id)
        else:
            return layer(x)

    def _bottleneck_forward(self, x, original_img, organ_id):
        if isinstance(self.bottleneck, UpConvBlockAttn):
            return self.bottleneck(x, original_img, organ_id)
        else:
            return self.bottleneck(x)

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


        x = pixel_values
        original_img = pixel_values
        feat_list = []

        # Padding
        pre_padding = (x.size(-1) % 2**self.depth != 0) or (
            x.size(-2) % 2**self.depth != 0
        ) or (x.size(-3) % 2**self.depth != 0)
        if pre_padding:
            x, pads = pad_to_2d(x, 2**self.depth)
            original_img, _ = pad_to_2d(original_img, 2**self.depth)

        # Encoder
        out, feat = self._enc_forward(self.encoder["0"], x, original_img, organ_id)
        feat_list.append(feat)

        for key in list(self.encoder.keys())[1:]:
            out, feat = self._enc_forward(
                self.encoder[key], out, original_img, organ_id
            )
            feat_list.append(feat)

        # Bottleneck
        out = self._bottleneck_forward(out, original_img, organ_id)
        out_bottleneck = out
        # Decoder
        for key in self.decoder:
            out = self._dec_forward(
                self.decoder[key],
                torch.cat((out, feat_list[int(key)]), dim=1),
                original_img,
                organ_id,
            )
            del feat_list[int(key)]

        # Output
        out = self.out_layer(out)

        if pre_padding:
            out = unpad_2d(out, pads).squeeze(1)

        # Loss
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
            loss = loss + (distill_loss_emb["loss"] + distill_loss_logits['loss'])/2

        return {
            "loss": loss,
            "logits": out,
            "labels": masks,
            "organ_id": organ_id,
            "organ_id_metric": organ_id_metric,
        }