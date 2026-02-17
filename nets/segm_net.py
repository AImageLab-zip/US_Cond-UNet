import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import wandb
from torchvision.transforms import v2
from nets.unet_base import BaseUnet


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
        self.embed = nn.Embedding(n_organs + 1, emb_dim)

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
        B = x.shape[0]
        mask = organ_id >= 0  # [B]
        q_org = self.embed(organ_id.clamp(min=0))  # [B, D] (dummy for unknown)
        q_img = self.embed(
            torch.tensor([self.embed.weight.shape[0] - 1], device=x.device).expand(B)
        )
        q = torch.where(mask[:, None], q_org, q_img)
        beta_gamma = self.mlp(q)  # (B, 2C)
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


class UNet2DFiLM(BaseUnet):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        n_organs: int,
        size: int = 32,
        depth: int = 3,
        *,
        film_start: int = 0,
        use_film: bool = True,
        film_embed: int = 64,
        distill: bool = False,
        distill_unet: bool = False,
        medsam_teacher_ckpt: str = "/work/phd_ultrasounds/UUSIC_new/checkpoints/medsam_unfreezed/model.safetensors",
        unet_teacher_ckpt: str = "/work/phd_ultrasounds/UUSIC_new/checkpoints/unet5_attn/model.safetensors",
        unet_teacher_kwargs: dict | None = None,
        use_selfaug: bool = False,
    ):
        """
        UNet with symmetric FiLM conditioning in encoder and decoder.
        """
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            n_organs=n_organs,
            size=size,
            depth=depth,
            film_start=film_start,
            use_film=use_film,
            film_embed=film_embed,
            distill=distill,
            distill_unet=distill_unet,
            medsam_teacher_ckpt=medsam_teacher_ckpt,
            unet_teacher_ckpt=unet_teacher_ckpt,
            unet_teacher_kwargs=unet_teacher_kwargs,
            use_selfaug=use_selfaug,
        )

    def _build_model(
        self,
        *,
        film_start: int = 0,
        use_film: bool = True,
        film_embed: int = 64,
        distill: bool = False,
        distill_unet: bool = False,
        medsam_teacher_ckpt: str = "/work/phd_ultrasounds/UUSIC_new/checkpoints/medsam_unfreezed/model.safetensors",
        unet_teacher_ckpt: str = "/work/phd_ultrasounds/UUSIC_new/checkpoints/unet5_attn/model.safetensors",
        unet_teacher_kwargs: dict | None = None,
        use_selfaug: bool = False,
        **kwargs,
    ):
        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected UNet2DFiLM kwargs: {unknown}")

        self.film_start = max(0, int(film_start))
        self.use_film = bool(use_film)
        self.film_embed = int(film_embed)
        self.use_selfaug = bool(use_selfaug)
        self.criterion = DiceBCELoss()

        # ---------------- Encoder ----------------
        self.encoder = nn.ModuleDict()

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

        for i in range(1, self.depth):
            in_ch = [self.size * (2**i), self.size * (2**i)]
            out_ch = [self.size * (2**i), self.size * (2 ** (i + 1))]
            key = str(i)

            if self.use_film and i >= self.film_start:
                self.encoder[key] = DownConvBlockFiLM(
                    in_ch,
                    out_ch,
                    n_organs=self.n_organs,
                    emb_dim=self.film_embed,
                )
            else:
                self.encoder[key] = DownConvBlock(in_ch, out_ch)

        # ---------------- Bottleneck ----------------
        if self.use_film:
            self.bottleneck = UpConvBlockFiLM(
                [self.size * (2**self.depth), self.size * (2**self.depth)],
                [self.size * (2**self.depth), self.size * (2 ** (self.depth + 1))],
                n_organs=self.n_organs,
                emb_dim=self.film_embed,
            )
        else:
            self.bottleneck = UpConvBlock(
                [self.size * (2**self.depth), self.size * (2**self.depth)],
                [self.size * (2**self.depth), self.size * (2 ** (self.depth + 1))],
            )

        # ---------------- Decoder ----------------
        self.decoder = nn.ModuleDict()

        for i in range(self.depth, 1, -1):
            use_film_at_level = self.use_film and (i - 1) >= self.film_start

            if use_film_at_level:
                self.decoder[str(i - 1)] = UpConvBlockFiLM(
                    [
                        self.size * (2 ** (i + 1)) + self.size * (2**i),
                        self.size * (2**i),
                    ],
                    [self.size * (2**i), self.size * (2**i)],
                    n_organs=self.n_organs,
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

        if self.use_film and 0 >= self.film_start:
            self.decoder["0"] = UpConvBlockFiLM(
                [self.size * 4 + self.size * 2, self.size * 2],
                [self.size * 2, self.size * 2],
                n_organs=self.n_organs,
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

        self._init_distillation(
            distill=distill,
            distill_unet=distill_unet,
            medsam_teacher_ckpt=medsam_teacher_ckpt,
            unet_teacher_ckpt=unet_teacher_ckpt,
            unet_teacher_kwargs=unet_teacher_kwargs,
        )
        
        self._init_selfaug(
            use_selfaug=self.use_selfaug
        )

    def _encode(
        self,
        layer,
        x,
        organ_id=None,
        forward_ctx=None,
    ):
        if isinstance(layer, DownConvBlockFiLM):
            return layer(x, organ_id)
        else:
            return layer(x)

    def _decode(
        self,
        layer,
        x,
        organ_id=None,
        forward_ctx=None,
    ):
        if isinstance(layer, UpConvBlockFiLM):
            return layer(x, organ_id)
        else:
            return layer(x)

    def _bottleneck(
        self,
        x,
        organ_id=None,
        forward_ctx=None,
    ):
        if isinstance(self.bottleneck, UpConvBlockFiLM):
            return self.bottleneck(x, organ_id)
        else:
            return self.bottleneck(x)

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
        self.eps = 1e-6

    def forward(self, logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:

        gt = gt.float()

        bce = F.binary_cross_entropy_with_logits(
            logits.squeeze(), gt.squeeze(), reduction="mean"
        )

        # Soft Dice loss
        probs = torch.sigmoid(logits)
        dims = tuple(range(2, probs.dim()))  # (H, W)  or (D,H,W)

        # per‑class Dice, per‑sample
        inter = (probs * gt).sum(dims) * 2
        union = probs.sum(dims) + gt.sum(dims)
        dice = 1 - (inter + self.eps) / (union + self.eps)  # [B, C]

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
        pixel_values_medsam=None,
        **kwargs,  # ignored, for peft compatibility
    ):
        batch_size = pixel_values.shape[0]

        # Get image embeddings
        image_embedding = self.image_encoder(pixel_values_medsam)  # (B, 256, 64, 64)

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

    def forward(self, student_logits, teacher_logits, tau=0.7):
        """
        Args:
            student_logits: (B, D) - student features
            teacher_logits: (B, D) - teacher features
        """
        # # Normalize features for stable training
        # student_norm = F.normalize(student_logits, p=2, dim=1)
        # teacher_norm = F.normalize(teacher_logits, p=2, dim=1)

        # # Cosine similarity loss (encourages directional alignment)
        # cosine_loss = (
        #     1 - F.cosine_similarity(student_logits, teacher_logits, dim=1).mean()
        # )

        # # MSE loss on normalized features (encourages magnitude alignment)
        # mse_loss = F.mse_loss(student_norm, teacher_norm)

        # # Combined loss
        # loss = self.alpha * cosine_loss + (1 - self.alpha) * mse_loss

        # # Optional L1 for sparsity
        # if self.use_l1:
        #     l1_loss = F.l1_loss(student_logits, teacher_logits)
        #     loss = loss + 0.1 * l1_loss

        B = student_logits.size(0)

        s = student_logits.flatten(start_dim=1)
        t = teacher_logits.flatten(start_dim=1).detach()

        s = F.normalize(s, dim=-1)
        t = F.normalize(t, dim=-1)

        d2 = (s[:, None, :] - t[None, :, :]).pow(2).sum(dim=-1)
        logits = -d2 / tau
        labels = torch.arange(B, device=s.device)

        loss = F.cross_entropy(logits, labels)
        return {
            "loss": loss,
        }


class MedSAMPrompt(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        freeze_image_encoder=True,
        predict_bboxes=False,
        n_organs=1,
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
        self.sparse_embeddings = nn.Parameter(torch.randn(1, 2, 256))
        self.dense_embeddings = nn.Embedding(n_organs + 1, 1 * 256 * 64 * 64)

        # self.sparse_embeddings = nn.ParameterDict(
        #     {
        #         str(id_): nn.Parameter(torch.randn(1, 2, 256))
        #         for id_ in set(organ_to_class_dict.values())
        #     }
        # )
        # self.dense_embeddings = nn.ParameterDict(
        #     {
        #         str(id_): nn.Parameter(torch.randn(1, 256, 64, 64))
        #         for id_ in set(organ_to_class_dict.values())
        #     }
        # )

    def forward(
        self,
        pixel_values,
        organ_id=None,
        labels=None,
        masks=None,
        bbox_coords=None,
        organ_id_metric=None,
        pixel_values_medsam=None,
        **kwargs,  # ignored, for peft compatibility
    ):
        batch_size = pixel_values.shape[0]
        B = pixel_values.shape[0]

        # Get image embeddings
        image_embedding = self.image_encoder(pixel_values_medsam)  # (B, 256, 64, 64)

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

        mask = organ_id >= 0  # [B]
        idx = torch.where(mask, organ_id, self.dense_embeddings.weight.shape[0] - 1)
        dense_embeddings = self.dense_embeddings(idx).view(B, 256, 64, 64)

        # Get positional encoding
        image_pe = self.prompt_encoder.get_dense_pe()  # (1, 256, 64, 64)

        # Decode mask
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=image_pe,  # (1, 256, 64, 64)
            sparse_prompt_embeddings=self.sparse_embeddings,  # (B, 2, 256)
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


# config = {"in_channels": 3,"num_classes": 1,"n_organs": 8,"size": 32,"depth": 5,"film_start": 0,"use_film": 1}
