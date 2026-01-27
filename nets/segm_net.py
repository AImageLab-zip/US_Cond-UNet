import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os, sys, wandb
from torchvision.transforms import v2
from utils.utils import organ_to_class_dict
from utils.paths import *


def pad_to_2d(x: torch.Tensor, stride: int):
    h, w = x.shape[-2:]

    new_h = h if h % stride == 0 else h + stride - (h % stride)
    new_w = w if w % stride == 0 else w + stride - (w % stride)

    top = (new_h - h) // 2
    bottom = (new_h - h) - top
    left = (new_w - w) // 2
    right = (new_w - w) - left

    pads = (left, right, top, bottom)
    x_pad = F.pad(x, pads, mode="constant", value=0)
    return x_pad, pads


def unpad_2d(x: torch.Tensor, pads):
    left, right, top, bottom = pads

    if top or bottom:
        end_h = -bottom if bottom > 0 else None
        x = x[:, :, top:end_h, :]

    if left or right:
        end_w = -right if right > 0 else None
        x = x[:, :, :, left:end_w]

    return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kwargs={"kernel_size": 3, "stride": 1, "padding": 1},
    ):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **conv_kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.block(x)


# Single encoder block
class DownConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: list,
        out_channels: list,
        conv_kwargs={"kernel_size": 3, "stride": 1, "padding": 1},
    ):
        super(DownConvBlock, self).__init__()

        assert len(in_channels) == len(
            out_channels
        ), f"in_channels length is {len(in_channels)} while out_channels is {len(out_channels)}"

        # Variable number of convolutional block in each layer, based on the in_channels and out_channels length
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(in_ch, out_ch, conv_kwargs)
                for in_ch, out_ch in zip(in_channels, out_channels)
            ]
        )

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        return self.pool(x), x


# Single decoder block
class UpConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: list,
        out_channels: list,
        up_conv=True,
        conv_kwargs={"kernel_size": 3, "stride": 1, "padding": 1},
        upconv_kwargs={"kernel_size": 2, "stride": 2},
    ):
        super(UpConvBlock, self).__init__()

        assert len(in_channels) == len(
            out_channels
        ), f"in_channels length is {len(in_channels)} while out_channels is {len(out_channels)}"

        # Variable number of convolutional block in each layer, based on the in_channels and out_channels length
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

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)

        if self.up_conv:
            return self.up_conv_op(x)
        else:
            return x


class FiLM2d(nn.Module):
    """
    Feature-wise Linear Modulation for a 2-D feature map.
    (γ, β) are generated from a learned embedding of organ_id.
    """

    def __init__(
        self,
        n_organs: int,
        in_channels: int,
        emb_dim: int | None = None,
        hidden: int | None = None,
    ):
        super().__init__()
        hidden = hidden or 2 * in_channels
        self.embed = nn.Embedding(n_organs, emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * in_channels),  # → [β‖γ]
        )

        # initialise so that FiLM starts as identity: γ≈1, β≈0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias[:in_channels], 0)  # β
        nn.init.constant_(self.mlp[-1].bias[in_channels:], 1)  # γ

    def forward(self, x: torch.Tensor, organ_id: torch.Tensor):
        """
        x : (B, C, H, W)
        organ_id : (B,) integer 0…n_organs-1
        """
        beta_gamma = self.mlp(self.embed(organ_id))  # (B, 2C)
        beta, gamma = beta_gamma.chunk(2, dim=-1)  # each (B, C)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class DownConvBlockFiLM(nn.Module):
    """
    Conv → FiLM → Conv → FiLM → Pool.
    Except for FiLM, the API and behaviour remain the same.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: list[int],
        n_organs: int,
        conv_kwargs={"kernel_size": 3, "stride": 1, "padding": 1},
        emb_dim: int = 64,
    ):
        super().__init__()
        assert len(in_channels) == len(
            out_channels
        ), f"in_channels length is {len(in_channels)} while out_channels is {len(out_channels)}"

        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(in_ch, out_ch, conv_kwargs)
                for in_ch, out_ch in zip(in_channels, out_channels)
            ]
        )

        self.film_blocks = nn.ModuleList(
            [
                FiLM2d(
                    n_organs=n_organs,
                    in_channels=out_ch,
                    emb_dim=emb_dim,
                )
                for out_ch in out_channels
            ]
        )

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor, organ_id: torch.Tensor):
        """
        organ_id : (B,) torch.long   (0 = breast, 1 = thyroid, …)
        """
        for conv, film in zip(self.conv_blocks, self.film_blocks):
            x = conv(x)  # usual conv-norm-ReLU
            x = film(x, organ_id)  # FiLM modulation
        return self.pool(x), x  # (downsampled, skip-connection)


class UpConvBlockFiLM(nn.Module):
    """
    Up-sampling block with FiLM conditioning.

    Conv → FiLM → Conv → FiLM → (optional) ConvTranspose2d
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: list[int],
        n_organs: int,
        up_conv: bool = True,
        conv_kwargs: dict = {"kernel_size": 3, "stride": 1, "padding": 1},
        upconv_kwargs: dict = {"kernel_size": 2, "stride": 2},
        emb_dim: int = 64,
    ):
        super().__init__()
        assert len(in_channels) == len(
            out_channels
        ), f"in_channels length is {len(in_channels)} while out_channels is {len(out_channels)}"

        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(in_ch, out_ch, conv_kwargs)
                for in_ch, out_ch in zip(in_channels, out_channels)
            ]
        )

        self.film_blocks = nn.ModuleList(
            [
                FiLM2d(
                    n_organs=n_organs,
                    in_channels=out_ch,
                    emb_dim=emb_dim,
                )
                for out_ch in out_channels
            ]
        )

        self.up_conv = up_conv
        if self.up_conv:
            self.up_conv_op = nn.ConvTranspose2d(
                out_channels[-1], out_channels[-1], **upconv_kwargs
            )

    def forward(self, x: torch.Tensor, organ_id: torch.Tensor):
        """
        x : (B, C, H, W)
        organ_id : (B,) long tensor - 0=breast, 1=thyroid, …
        """
        for conv, film in zip(self.conv_blocks, self.film_blocks):
            x = conv(x)
            x = film(x, organ_id)

        if self.up_conv:
            x = self.up_conv_op(x)

        return x


class UNet2DFiLM(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        n_organs,
        size=32,
        depth=3,
        film_start: int = 0,
        use_film=True,
        film_embed=64,
        distill=False,
    ):
        """
        UNet with symmetric FiLM conditioning in encoder and decoder.

        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            n_organs: Number of organ types for FiLM conditioning
            size: Base number of channels
            depth: Number of encoder/decoder levels
            film_start: 0-based index of the encoder level where FiLM starts
                       0  -> FiLM from the first encoder block
                       k  -> encoder blocks [0..k-1] plain, [k..depth-1] FiLM
                       >= depth -> no FiLM in encoder
            use_film: If False, disables FiLM completely (uses plain Conv blocks)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = num_classes
        self.size = size
        self.depth = depth
        self.film_start = max(0, int(film_start))
        self.use_film = use_film
        self.n_organs = n_organs
        self.film_embed = film_embed
        self.distill = distill
        self.criterion = DiceBCELoss()

        # ---------------- Encoder ----------------
        self.encoder = nn.ModuleDict()

        # First encoder block
        if self.use_film and 0 >= self.film_start:
            self.encoder["0"] = DownConvBlockFiLM(
                [self.in_channels, self.size],
                [self.size, self.size * 2],
                n_organs=self.n_organs,
                emb_dim=self.film_embed,
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

            if self.use_film and i >= self.film_start:
                self.encoder[key] = DownConvBlockFiLM(
                    in_ch,
                    out_ch,
                    n_organs=n_organs,
                    emb_dim=self.film_embed,
                )
            else:
                self.encoder[key] = DownConvBlock(in_ch, out_ch)

        # ---------------- Bottleneck ----------------
        if self.use_film:
            self.bottleneck = UpConvBlockFiLM(
                [self.size * (2**self.depth), self.size * (2**self.depth)],
                [self.size * (2**self.depth), self.size * (2 ** (self.depth + 1))],
                n_organs=n_organs,
                emb_dim=self.film_embed,
            )
        else:
            self.bottleneck = UpConvBlock(
                [self.size * (2**self.depth), self.size * (2**self.depth)],
                [self.size * (2**self.depth), self.size * (2 ** (self.depth + 1))],
            )

        # ---------------- Decoder (symmetric FiLM usage) ----------------
        self.decoder = nn.ModuleDict()

        for i in range(self.depth, 1, -1):
            # Determine if this decoder level should use FiLM
            # Mirror the encoder: if encoder[i-1] uses FiLM, decoder[i-1] uses FiLM
            use_film_at_level = self.use_film and (i - 1) >= self.film_start

            if use_film_at_level:
                self.decoder[str(i - 1)] = UpConvBlockFiLM(
                    [
                        self.size * (2 ** (i + 1)) + self.size * (2**i),
                        self.size * (2**i),
                    ],
                    [self.size * (2**i), self.size * (2**i)],
                    n_organs=n_organs,
                    emb_dim=self.film_embed,
                )
            else:
                self.decoder[str(i - 1)] = UpConvBlock(
                    [
                        self.size * (2 ** (i + 1)) + self.size * (2**i),
                        self.size * (2**i),
                    ],
                    [self.size * (2**i), self.size * (2**i)],
                )

        # Final decoder block (level 0)
        if self.use_film and 0 >= self.film_start:
            self.decoder["0"] = UpConvBlockFiLM(
                [self.size * 4 + self.size * 2, self.size * 2],
                [self.size * 2, self.size * 2],
                n_organs=n_organs,
                up_conv=False,
                emb_dim=self.film_embed,
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
            from segment_anything import sam_model_registry
            from copy import deepcopy
            from nets.segm_net import UNet2DFiLM, MedSAM, MedSAMPrompt
            from safetensors.torch import load_file

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

    def _enc_forward(self, layer, x, organ_id):
        """Helper to call encoder blocks with or without FiLM"""
        if isinstance(layer, DownConvBlockFiLM):
            return layer(x, organ_id)
        else:
            return layer(x)

    def _dec_forward(self, layer, x, organ_id):
        """Helper to call decoder blocks with or without FiLM"""
        if isinstance(layer, UpConvBlockFiLM):
            return layer(x, organ_id)
        else:
            return layer(x)

    def _bottleneck_forward(self, x, organ_id):
        """Helper to call bottleneck with or without FiLM"""
        if isinstance(self.bottleneck, UpConvBlockFiLM):
            return self.bottleneck(x, organ_id)
        else:
            return self.bottleneck(x)

    def encode(self, x, organ_id):
        """Encoder pass with skip connections"""
        feat_list = []

        # Padding if needed
        pre_padding = (
            (x.size(-1) % 2**self.depth != 0)
            or (x.size(-2) % 2**self.depth != 0)
            or (x.size(-3) % 2**self.depth != 0)
        )
        if pre_padding:
            x, pads = pad_to_2d(x, 2**self.depth)
        else:
            pads = None

        # First encoder block
        out, feat = self._enc_forward(self.encoder["0"], x, organ_id)
        feat_list.append(feat)

        # Remaining encoder blocks
        for key in list(self.encoder.keys())[1:]:
            out, feat = self._enc_forward(self.encoder[key], out, organ_id)
            feat_list.append(feat)

        # Bottleneck
        out = self._bottleneck_forward(out, organ_id)

        return out, feat_list, pads

    def decode(self, x, organ_id, out, feat_list, pads):
        """Decoder pass with skip connections"""
        # Decoder blocks
        for key in self.decoder:
            out = self._dec_forward(
                self.decoder[key],
                torch.cat((out, feat_list[int(key)]), dim=1),
                organ_id,
            )
            del feat_list[int(key)]

        # Output layer
        out = self.out_layer(out)

        # Remove padding if it was added
        if pads is not None:
            out = unpad_2d(out, pads)

        return out

    def forward(
        self,
        pixel_values,
        organ_id=None,
        labels=None,
        masks=None,
        bbox_coords=None,
        organ_id_metric=None,
        **kwargs,  # ignored, for peft compatibility
    ):
        """
        Full forward pass through the network.

        Args:
            pixel_values: Input images (B, C, H, W)
            organ_id: Organ type IDs for FiLM conditioning (B,)
            labels: Ground truth labels (optional)
            masks: Ground truth masks (B, H, W)
            bbox_coords: Bounding box coordinates (optional)
        """
        x = pixel_values
        feat_list = []

        # Check if padding is needed
        pre_padding = (
            (x.size(-1) % 2**self.depth != 0)
            or (x.size(-2) % 2**self.depth != 0)
            or (x.size(-3) % 2**self.depth != 0)
        )
        if pre_padding:
            x, pads = pad_to_2d(x, 2**self.depth)

        # Encoder
        out, feat = self._enc_forward(self.encoder["0"], x, organ_id)
        feat_list.append(feat)

        for key in list(self.encoder.keys())[1:]:
            out, feat = self._enc_forward(self.encoder[key], out, organ_id)
            feat_list.append(feat)

        # Bottleneck
        out = self._bottleneck_forward(out, organ_id)
        out_bottleneck = out
        # Decoder
        for key in self.decoder:
            out = self._dec_forward(
                self.decoder[key],
                torch.cat((out, feat_list[int(key)]), dim=1),
                organ_id,
            )
            del feat_list[int(key)]

        # Output
        out = self.out_layer(out)

        if pre_padding:
            out = unpad_2d(out, pads).squeeze(1)

        # Calculate loss if masks are provided
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

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        film_status = "enabled" if self.use_film else "disabled"
        film_range = f"from level {self.film_start}" if self.use_film else "N/A"
        return (
            super().__str__() + f"\nTrainable parameters: {params}"
            f"\nFiLM: {film_status} ({film_range})"
        )


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.eps = 1e-7  # Safe for bf16, will be adjusted for fp16

    def forward(self, logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # Adjust epsilon based on dtype
        eps = 1e-4 if logits.dtype == torch.float16 else self.eps

        # Create mask for valid samples (not all -100)
        valid_mask = (gt != -100).any(dim=tuple(range(1, gt.dim())))  # [B]

        # If no valid samples, return zero loss
        if not valid_mask.any():
            return logits.sum() * 0.0

        # Filter out invalid samples
        logits = logits[valid_mask]
        gt = gt[valid_mask]

        # Replace -100 with 0 for any remaining -100 values (partial masks)
        gt = gt.clone()
        gt[gt == -100] = 0
        gt = gt.float()

        # BCE loss
        bce = F.binary_cross_entropy_with_logits(
            logits.squeeze(), gt.squeeze(), reduction="mean"
        )

        # Soft Dice loss
        probs = torch.sigmoid(logits)
        dims = tuple(range(2, probs.dim()))  # (H, W) or (D, H, W)

        # per-class Dice, per-sample
        inter = (probs * gt).sum(dims) * 2
        union = probs.sum(dims) + gt.sum(dims)
        dice = 1 - (inter + eps) / (union + eps)  # [B, C]

        dice = dice.mean()

        loss = self.dice_weight * dice + self.bce_weight * bce
        return loss


class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        freeze_image_encoder=True,
        predict_bboxes=False,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.criterion = DiceBCELoss()
        self.freeze_image_encoder = freeze_image_encoder
        self.predict_bboxes = predict_bboxes

        if self.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        # Classification head
        self.multi_cls = nn.Sequential(nn.Linear(256 * 64 * 64, 10))

        # Bounding box regression head
        self.bbox_regr = nn.Sequential(nn.Linear(256 * 64 * 64, 4))

        # Learnable prompt embeddings (no input required)
        self.learned_sparse_embeddings = nn.Parameter(
            torch.randn(1, 2, 256)  # (batch, num_tokens, embed_dim)
        )
        self.learned_dense_embeddings = nn.Parameter(
            torch.randn(1, 256, 64, 64)  # (batch, embed_dim, H, W)
        )

    def forward(
        self,
        pixel_values,
        organ_id=None,
        labels=None,
        masks=None,
        bbox_coords=None,
        organ_id_metric=None,
    ):
        batch_size = pixel_values.shape[0]

        # Get image embeddings
        image_embedding = self.image_encoder(pixel_values)  # (B, 256, 64, 64)

        # Classification output
        emb_flattened = torch.flatten(image_embedding, 1)
        multi_cls_out = self.multi_cls(emb_flattened)

        # Bounding box output (if enabled)
        if self.predict_bboxes:
            bbox_out = self.bbox_regr(emb_flattened)
        else:
            bbox_out = None

        # Expand learned embeddings to batch size
        sparse_embeddings = self.learned_sparse_embeddings
        dense_embeddings = self.learned_dense_embeddings

        # Get positional encoding
        image_pe = self.prompt_encoder.get_dense_pe()  # (1, 256, 64, 64)

        # Decode mask
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=image_pe,  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )  # (B, 1, 256, 256)
        if masks is not None:
            loss = self.criterion(
                low_res_masks.squeeze(1), v2.functional.resize(masks, (256, 256))
            )
        else:
            loss = 0.0

        return {
            "loss": loss,
            "logits": low_res_masks.squeeze(1),
            "labels": masks,
            "organ_id": organ_id,
            "organ_id_metric": organ_id_metric,
        }


class DistillationLoss(nn.Module):
    """
    Improved distillation loss with multiple components:
    - Cosine similarity loss (direction alignment)
    - MSE loss (magnitude alignment)
    - Optional L1 loss (sparsity)
    """

    def __init__(self, temperature=3.0, alpha=0.5, use_l1=False):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight between cosine and MSE
        self.use_l1 = use_l1

    def forward(self, student_logits, teacher_logits):
        """
        Args:
            student_logits: (B, D) - student features
            teacher_logits: (B, D) - teacher features
        """
        # Normalize features for stable training
        student_norm = F.normalize(student_logits, p=2, dim=1)
        teacher_norm = F.normalize(teacher_logits, p=2, dim=1)

        # Cosine similarity loss (encourages directional alignment)
        cosine_loss = (
            1 - F.cosine_similarity(student_logits, teacher_logits, dim=1).mean()
        )

        # MSE loss on normalized features (encourages magnitude alignment)
        mse_loss = F.mse_loss(student_norm, teacher_norm)

        # Combined loss
        loss = self.alpha * cosine_loss + (1 - self.alpha) * mse_loss

        # Optional L1 for sparsity
        if self.use_l1:
            l1_loss = F.l1_loss(student_logits, teacher_logits)
            loss = loss + 0.1 * l1_loss

        return {
            "loss": loss,
            "cosine_loss": cosine_loss.item(),
            "mse_loss": mse_loss.item(),
        }


class UNet2DFiLMDistillation(nn.Module):
    """
    Improved distillation wrapper with better feature alignment.
    """

    def __init__(
        self,
        student_config,
        teacher_model=None,
        temperature=3.0,
        extract_features=True,
        use_patch_tokens=False,  # New: use DINOv3 patch tokens instead of CLS
    ):
        super().__init__()

        # Initialize student model
        self.student = UNet2DFiLM(**student_config)

        # Calculate student bottleneck size
        # Bottleneck output has size * 2^(depth+1) channels
        # depth=5: 32 * 2^6 = 2048 channels
        # depth=3: 32 * 2^4 = 512 channels
        student_bottleneck_channels = student_config["size"] * (
            2 ** (student_config["depth"] + 1)
        )

        self.use_patch_tokens = use_patch_tokens

        if use_patch_tokens:
            # DINOv3 vit-huge/16 with 512x512 input -> 32x32 patch tokens
            # Each token is 1280-dim
            # We'll use spatial features from student
            teacher_dim = 1280
            # Keep spatial, just reduce channels
            self.adaptive_pool = nn.AdaptiveAvgPool2d((32, 32))  # Match DINOv3 patches
            self.adapter = nn.Sequential(
                nn.Conv2d(student_bottleneck_channels, 512, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, teacher_dim, 1),
            )
        else:
            # Use global features (CLS token from DINOv3)
            teacher_dim = 1280
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.adapter = nn.Sequential(
                nn.Linear(student_bottleneck_channels, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, teacher_dim),
            )

        # Set up teacher model
        self.teacher = teacher_model
        if self.teacher is not None:
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.teacher.eval()

        # Improved distillation loss
        self.distillation_loss = DistillationLoss(temperature, alpha=0.7)
        self.extract_features = extract_features

    def set_teacher(self, teacher_model):
        """Set or update the teacher model after initialization"""
        self.teacher = teacher_model
        if self.teacher is not None:
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.teacher.eval()

    def train(self, mode=True):
        """Override train() to keep teacher in eval mode"""
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def forward(
        self,
        pixel_values,
        organ_id=None,
        labels=None,
        masks=None,
        bbox_coords=None,
        return_teacher=False,
    ):
        """
        Forward pass with improved knowledge distillation.
        """
        # Student forward pass
        stud_out = self.student.encode(pixel_values, organ_id)
        stud_bottleneck = stud_out[0]  # (B, C, H, W)

        if self.use_patch_tokens:
            # Spatial distillation
            pooled = self.adaptive_pool(stud_bottleneck)  # (B, C, 32, 32)
            student_features = self.adapter(pooled)  # (B, 1280, 32, 32)
            # Flatten spatial dimensions: (B, 1280, 1024)
            student_logits = student_features.flatten(2).transpose(1, 2)
            student_logits = student_logits.mean(
                dim=1
            )  # Average over patches (B, 1280)
        else:
            # Global distillation
            pooled = self.adaptive_pool(stud_bottleneck)  # (B, C, 1, 1)
            pooled = pooled.flatten(1)  # (B, C)
            student_logits = self.adapter(pooled)  # (B, 1280)

        # Teacher forward pass (no gradients)
        teacher_logits = None
        if self.teacher is not None:
            with torch.no_grad():
                if self.use_patch_tokens:
                    # Get patch tokens from DINOv3
                    teacher_output = self.teacher.forward_features(pixel_values)
                    # Average patch tokens (excluding CLS)
                    teacher_logits = teacher_output["x_norm_patchtokens"].mean(
                        dim=1
                    )  # (B, 1280)
                else:
                    # Get CLS token from DINOv3
                    teacher_logits = self.teacher(pixel_values)  # (B, 1280)

        # Calculate distillation loss
        if self.teacher is not None and teacher_logits is not None:
            loss_dict = self.distillation_loss(student_logits, teacher_logits)
            loss = loss_dict["loss"]
        else:
            raise ValueError("Teacher model is required for distillation training")

        output = {
            "loss": loss,
            "logits": student_logits,
            "labels": masks,
            "organ_id": organ_id,
            "cosine_loss": loss_dict.get("cosine_loss", 0),
            "mse_loss": loss_dict.get("mse_loss", 0),
        }

        return output

    def __str__(self):
        student_params = sum(
            p.numel() for p in self.student.parameters() if p.requires_grad
        )
        adapter_params = sum(
            p.numel() for p in self.adapter.parameters() if p.requires_grad
        )
        teacher_params = (
            sum(p.numel() for p in self.teacher.parameters())
            if self.teacher is not None
            else 0
        )

        return (
            f"UNet2DFiLMDistillation(\n"
            f"  Student parameters: {student_params:,}\n"
            f"  Adapter parameters: {adapter_params:,}\n"
            f"  Teacher parameters: {teacher_params:,}\n"
            f"  Temperature: {self.distillation_loss.temperature}\n"
            f"  Mode: {'Patch tokens' if self.use_patch_tokens else 'CLS token'}\n"
            f")"
        )


class MedSAMPrompt(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        freeze_image_encoder=True,
        predict_bboxes=False,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.criterion = DiceBCELoss()
        self.freeze_image_encoder = freeze_image_encoder
        self.predict_bboxes = predict_bboxes

        if self.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        # Classification head
        self.multi_cls = nn.Sequential(nn.Linear(256 * 64 * 64, 10))

        # Bounding box regression head
        self.bbox_regr = nn.Sequential(nn.Linear(256 * 64 * 64, 4))

        # Learnable prompt embeddings (no input required)
        self.sparse_embeddings = nn.ParameterDict(
            {
                str(id_): nn.Parameter(torch.randn(1, 2, 256))
                for id_ in set(organ_to_class_dict.values())
            }
        )
        self.dense_embeddings = nn.ParameterDict(
            {
                str(id_): nn.Parameter(torch.randn(1, 256, 64, 64))
                for id_ in set(organ_to_class_dict.values())
            }
        )

    def forward(
        self, pixel_values, organ_id=None, labels=None, masks=None, bbox_coords=None
    ):
        batch_size = pixel_values.shape[0]

        # Get image embeddings
        image_embedding = self.image_encoder(pixel_values)  # (B, 256, 64, 64)

        # Classification output
        emb_flattened = torch.flatten(image_embedding, 1)
        multi_cls_out = self.multi_cls(emb_flattened)

        # Bounding box output (if enabled)
        if self.predict_bboxes:
            bbox_out = self.bbox_regr(emb_flattened)
        else:
            bbox_out = None

        if organ_id is None:
            raise ValueError("organ_id must be provided for selecting embeddings.")

        sparse_emb_list, dense_emb_list = [], []
        for oid in organ_id:

            key = str(int(oid.item()))
            sparse_emb_list.append(self.sparse_embeddings[key])
            dense_emb_list.append(self.dense_embeddings[key])

        # Stack embeddings for batch
        sparse_embeddings = torch.cat(sparse_emb_list, dim=0)  # (B, 2, 256)
        dense_embeddings = torch.cat(dense_emb_list, dim=0)  # (B, 256, 64, 64)

        # Get positional encoding
        image_pe = self.prompt_encoder.get_dense_pe()  # (1, 256, 64, 64)

        # Decode mask
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=image_pe,  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )  # (B, 1, 256, 256)
        if masks is not None:
            loss = self.criterion(
                low_res_masks.squeeze(1), v2.functional.resize(masks, (256, 256))
            )
        else:
            loss = 0.0

        return {
            "loss": loss,
            "logits": low_res_masks.squeeze(1),
            "labels": masks,
            "organ_id": organ_id,
        }


# config = {"in_channels": 3,"num_classes": 1,"n_organs": 8,"size": 32,"depth": 5,"film_start": 0,"use_film": 1}
