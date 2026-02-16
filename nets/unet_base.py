from copy import deepcopy
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from torchvision.transforms import v2
import wandb


class BaseUnet(nn.Module, ABC):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        n_organs: int,
        size: int = 32,
        depth: int = 3,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = num_classes
        self.n_organs = n_organs
        self.size = size
        self.depth = depth
        self.distill = False
        self.distill_unet = False
        self.distill_model = None
        self.distill_adapter = None
        self.distill_loss = None
        self._build_model(**kwargs)

    @staticmethod
    def _pad_to_2d(x: torch.Tensor, stride: int):
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

    @staticmethod
    def _unpad_2d(x: torch.Tensor, pads):
        left, right, top, bottom = pads

        if top or bottom:
            end_h = -bottom if bottom > 0 else None
            x = x[:, :, top:end_h, :]

        if left or right:
            end_w = -right if right > 0 else None
            x = x[:, :, :, left:end_w]

        return x

    @abstractmethod
    def _build_model(self, **kwargs):
        pass

    def _prepare_forward(
        self,
        *,
        pixel_values: torch.Tensor,
        organ_id: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        return {}

    @abstractmethod
    def _encode(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        organ_id: torch.Tensor | None = None,
        forward_ctx: dict | None = None,
    ):
        pass

    @abstractmethod
    def _bottleneck(
        self,
        x: torch.Tensor,
        organ_id: torch.Tensor | None = None,
        forward_ctx: dict | None = None,
    ):
        pass

    @abstractmethod
    def _decode(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        organ_id: torch.Tensor | None = None,
        forward_ctx: dict | None = None,
    ):
        pass

    def encode(
        self,
        x: torch.Tensor,
        organ_id: torch.Tensor | None = None,
        forward_ctx: dict | None = None,
    ):
        feat_list = []
        pads = None

        pre_padding = (
            (x.size(-1) % 2**self.depth != 0)
            or (x.size(-2) % 2**self.depth != 0)
            or (x.size(-3) % 2**self.depth != 0)
        )
        if pre_padding:
            x, pads = self._pad_to_2d(x, 2**self.depth)

        out, feat = self._encode(
            self.encoder["0"], x, organ_id=organ_id, forward_ctx=forward_ctx
        )
        feat_list.append(feat)

        for key in list(self.encoder.keys())[1:]:
            out, feat = self._encode(
                self.encoder[key], out, organ_id=organ_id, forward_ctx=forward_ctx
            )
            feat_list.append(feat)

        out = self._bottleneck(out, organ_id=organ_id, forward_ctx=forward_ctx)
        return out, feat_list, pads

    def decode(
        self,
        out: torch.Tensor,
        feat_list: list[torch.Tensor],
        pads,
        organ_id: torch.Tensor | None = None,
        forward_ctx: dict | None = None,
    ):
        for key in self.decoder:
            out = self._decode(
                self.decoder[key],
                torch.cat((out, feat_list[int(key)]), dim=1),
                organ_id=organ_id,
                forward_ctx=forward_ctx,
            )
            del feat_list[int(key)]

        out = self.out_layer(out)
        if pads is not None:
            out = self._unpad_2d(out, pads).squeeze(1)
        return out

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
        return loss

    def _init_distillation(
        self,
        *,
        distill: bool = False,
        distill_unet: bool = False,
        medsam_teacher_ckpt: str = "/work/phd_ultrasounds/UUSIC_new/checkpoints/medsam_unfreezed/model.safetensors",
        unet_teacher_ckpt: str = "/work/phd_ultrasounds/UUSIC_new/checkpoints/unet5_attn_distilled/model.safetensors",
        unet_teacher_kwargs: dict | None = None,
    ):
        if distill and distill_unet:
            raise ValueError("distill and distill_unet cannot both be enabled.")

        self.distill = bool(distill)
        self.distill_unet = bool(distill_unet)
        self.distill_model = None
        self.distill_adapter = None
        self.distill_loss = None

        if not self.distill and not self.distill_unet:
            return

        from nets.segm_net import DistillationLoss, MedSAM

        student_channels = (2048 // (32 // self.size)) // (2 ** (5 - self.depth))

        if self.distill:
            self.distill_adapter = nn.Conv2d(student_channels, 256, kernel_size=1)
        else:
            self.distill_adapter = nn.Conv2d(student_channels, 2048, kernel_size=1)

        if self.distill:
            from segment_anything import sam_model_registry
            from utils.paths import MEDSAM_BASE_WEIGHTS

            sam_model = sam_model_registry["vit_b"](checkpoint=MEDSAM_BASE_WEIGHTS)
            self.distill_model = MedSAM(
                image_encoder=deepcopy(sam_model.image_encoder),
                mask_decoder=deepcopy(sam_model.mask_decoder),
                prompt_encoder=deepcopy(sam_model.prompt_encoder),
                predict_bboxes=True,
                freeze_image_encoder=0,
            )
            state_dict = load_file(medsam_teacher_ckpt)
            load_result = self.distill_model.load_state_dict(state_dict)
            print(f"Loaded MedSam teacher model and loaded weights:\n{load_result}")
        else:
            from nets.unet_attn import UNet2DAttn

            teacher_kwargs = {
                "in_channels": 3,
                "num_classes": 1,
                "n_organs": 10,
                "size": 32,
                "depth": 5,
                "attn_start": 0,
                "use_attn": True,
                "img_size": 512,
                "patch_size": 8,
                "emb_dim": 768,
                "n_heads": 8,
                "distill": False,
                "distill_unet": False,
                "use_dwt": False,
                "wavelet": "haar",
                "use_shape": False,
                "shape_res": 64,
            }
            if unet_teacher_kwargs is not None:
                teacher_kwargs.update(unet_teacher_kwargs)

            self.distill_model = UNet2DAttn(**teacher_kwargs)
            state_dict = load_file(unet_teacher_ckpt)
            state_dict = {k: v for k, v in state_dict.items() if "distill" not in k}
            load_result = self.distill_model.load_state_dict(state_dict)
            print(f"Loaded UNet teacher model and loaded weights:\n{load_result}")

        for p in self.distill_model.parameters():
            p.requires_grad = False
        self.distill_model.eval()
        self.distill_loss = DistillationLoss()

    def _forward_distillation(
        self,
        *,
        student_logits: torch.Tensor,
        student_bottleneck: torch.Tensor,
        pixel_values: torch.Tensor,
        organ_id: torch.Tensor | None = None,
        pixel_values_medsam: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if not self.distill and not self.distill_unet:
            return None

        if self.distill:
            if pixel_values_medsam is None:
                raise ValueError("pixel_values_medsam is required when distill=True.")

            with torch.no_grad():
                up_pixel_values = v2.functional.resize(
                    pixel_values_medsam, 1024, v2.InterpolationMode.BICUBIC
                )
                image_embedding = self.distill_model.image_encoder(up_pixel_values)

            student_resized = F.interpolate(
                student_bottleneck,
                size=(image_embedding.shape[-1], image_embedding.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            up_feat = self.distill_adapter(student_resized)
            distill_loss_emb = self.distill_loss(
                student_logits=up_feat,
                teacher_logits=image_embedding.detach(),
            )

            if wandb.run is not None:
                wandb.log(
                    {
                        "distill_loss_emb": distill_loss_emb["loss"].item(),
                    },
                    commit=False,
                )

            return distill_loss_emb["loss"]

        with torch.no_grad():
            forward_ctx = self._prepare_forward(
                pixel_values=pixel_values,
                organ_id=organ_id,
            )

            teacher_embedding, _, _ = self.encode(
                pixel_values, organ_id=organ_id, forward_ctx=forward_ctx
            )

        student_resized = F.interpolate(
                student_bottleneck,
                size=(teacher_embedding.shape[-1], teacher_embedding.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
        up_feat = self.distill_adapter(student_resized)
        distill_loss_emb = self.distill_loss(
            student_logits=student_bottleneck,
            teacher_logits=teacher_embedding.detach(),
        )
        if wandb.run is not None:
            wandb.log(
                {"distill_loss_logits": distill_loss_emb["loss"].item()},
                commit=False,
            )
        return distill_loss_emb["loss"]

    def forward(
        self,
        pixel_values,
        organ_id=None,
        labels=None,
        masks=None,
        bbox_coords=None,
        organ_id_metric=None,
        teacher_embedding=None,
        teacher_mask=None,
        pixel_values_medsam=None,
        **kwargs,
    ):
        forward_ctx = self._prepare_forward(
            pixel_values=pixel_values,
            organ_id=organ_id,
            masks=masks,
            **kwargs,
        )

        out_bottleneck, feat_list, pads = self.encode(
            pixel_values, organ_id=organ_id, forward_ctx=forward_ctx
        )
        out = self.decode(
            out_bottleneck,
            feat_list,
            pads,
            organ_id=organ_id,
            forward_ctx=forward_ctx,
        )

        if masks is not None:
            loss = self.criterion(out, masks)
        else:
            loss = 0.0

        distill_loss = self._forward_distillation(
            student_logits=out,
            student_bottleneck=out_bottleneck,
            pixel_values=pixel_values,
            organ_id=organ_id,
            pixel_values_medsam=pixel_values_medsam,
        )
        if distill_loss is not None:
            loss = loss + distill_loss

        loss = self._apply_auxiliary_losses(
            loss=loss,
            logits=out,
            masks=masks,
            organ_id=organ_id,
            forward_ctx=forward_ctx,
            pixel_values=pixel_values,
            **kwargs,
        )

        return {
            "loss": loss,
            "logits": out,
            "labels": masks,
            "organ_id": organ_id,
            "organ_id_metric": organ_id_metric,
        }
