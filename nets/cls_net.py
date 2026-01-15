from types import NoneType
import torch
from torch import nn
from torchvision import models
import wandb
from utils.utils import organ_to_class_dict, multi_cls_labels_dict


class FiLM2d(nn.Module):
    """
    Feature-wise Linear Modulation for a 2-D feature map.
    (γ, β) are generated from a learned embedding of organ_id.
    """

    def __init__(
        self,
        n_organs: int,
        in_channels: int,
        emb_dim: int = 64,
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


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class OmniClsCBAM(nn.Module):
    def __init__(
        self,
        resnet_size="18",
        use_cbam=True,
        use_film=False,
        n_organs=len(organ_to_class_dict.keys()),
        num_classes=12,
        predict_bboxes=False,
        mlp_organ=False,
    ):
        """
        Simplified OmniClsCBAM with optional CBAM and FiLM modules.

        Args:
            resnet_size (str): ResNet architecture size ('18', '34', '50', '101', '152')
            use_cbam (bool): Whether to use CBAM attention modules
            use_film (bool): Whether to use FiLM conditioning layers
            n_organs (int): Number of different organs for FiLM conditioning
            num_classes (int): Number of output classes
        """
        super().__init__()

        self.use_cbam = use_cbam
        self.use_film = use_film
        self.predict_bboxes = predict_bboxes
        self.mlp_organ = mlp_organ
        self.criterion = MultiLoss(
            device=(
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            ),
            wandb_logging=True,
            predict_bboxes=self.predict_bboxes,
            normalize_factor=256,
        )

        resnet_map = {
            "18": models.resnet18,
            "34": models.resnet34,
            "50": models.resnet50,
            "101": models.resnet101,
            "152": models.resnet152,
        }
        if resnet_size not in resnet_map:
            raise ValueError(
                f"Unsupported resnet_size '{resnet_size}'. Choose from {list(resnet_map.keys())}."
            )

        backbone_model = resnet_map[resnet_size](weights="DEFAULT")

        # Helper function to get the output channels of a layer
        def get_layer_channels(layer):
            last_block = layer[-1]
            if hasattr(last_block, "conv3"):  # Bottleneck
                return last_block.conv3.out_channels
            else:  # BasicBlock
                return last_block.conv2.out_channels

        # Build backbone components
        self.stem = nn.Sequential(
            backbone_model.conv1,
            backbone_model.bn1,
            backbone_model.relu,
            backbone_model.maxpool,
        )

        self.layer1 = backbone_model.layer1
        self.layer2 = backbone_model.layer2
        self.layer3 = backbone_model.layer3
        self.layer4 = backbone_model.layer4
        self.avgpool = backbone_model.avgpool

        # Add CBAM modules if requested
        if use_cbam:
            self.cbam1 = CBAM(get_layer_channels(backbone_model.layer1))
            self.cbam2 = CBAM(get_layer_channels(backbone_model.layer2))
            self.cbam3 = CBAM(get_layer_channels(backbone_model.layer3))
            self.cbam4 = CBAM(get_layer_channels(backbone_model.layer4))

        # Add FiLM modules if requested
        if use_film:
            self.film1 = FiLM2d(n_organs, get_layer_channels(backbone_model.layer1))
            self.film2 = FiLM2d(n_organs, get_layer_channels(backbone_model.layer2))
            self.film3 = FiLM2d(n_organs, get_layer_channels(backbone_model.layer3))
            self.film4 = FiLM2d(n_organs, get_layer_channels(backbone_model.layer4))

        self.embed_dim = backbone_model.fc.in_features + (64 if self.mlp_organ else 0)
        self.organ_mlp = nn.Sequential(
            nn.Embedding(n_organs, 64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),  # Output dim to concat, e.g., 64
        )
        self.multi_cls = nn.Linear(self.embed_dim, num_classes)  # Adjust in_features
        self.bbox_regr = nn.Linear(self.embed_dim, 4)  # Adjust similarly
        self.dropout = nn.Dropout(p=0.2)

        self.initialize_weights_()

    def initialize_weights_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def freeze_backbone(self):
        """
        Freeze all backbone parameters (stem, layer1-4, avgpool).
        """
        for param in self.stem.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
        for param in self.layer3.parameters():
            param.requires_grad = False
        for param in self.layer4.parameters():
            param.requires_grad = False
        for param in self.avgpool.parameters():
            param.requires_grad = False
        print("Freezed backbone!")

    def unfreeze_backbone(self):
        """
        Unfreeze all backbone parameters (stem, layer1-4, avgpool).
        """
        for param in self.stem.parameters():
            param.requires_grad = True
        for param in self.layer1.parameters():
            param.requires_grad = True
        for param in self.layer2.parameters():
            param.requires_grad = True
        for param in self.layer3.parameters():
            param.requires_grad = True
        for param in self.layer4.parameters():
            param.requires_grad = True
        for param in self.avgpool.parameters():
            param.requires_grad = True
        print("Unfreezed backbone!")

    def forward(
        self, pixel_values, organ_id=None, labels=None, masks=None, bbox_coords=None
    ):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input images (B, C, H, W)
            organ_id (torch.Tensor, optional): Organ IDs for FiLM conditioning (B,)

        Returns:
            tuple: (multi_cls_out, bbox_out)
        """
        x = pixel_values
        x = self.get_feat(x, organ_id)

        # organ_emb = self.organ_embed(organ_id)  # (B, 64)
        if self.mlp_organ:
            organ_emb = self.organ_mlp(organ_id)  # (B, 64)
            x = torch.cat([x, organ_emb], dim=1)  # (B, embed_dim + 64)
        x = self.dropout(x)

        multi_cls_out = self.multi_cls(x)
        bbox_out = self.bbox_regr(x)

        loss = self.criterion(multi_cls_out, bbox_out, labels, bbox_coords, organ_id)

        return {"loss": loss, "logits": multi_cls_out, "organ_id": organ_id}

    def get_feat(self, x, organ_id=None):
        if self.use_film and organ_id is None:
            raise ValueError("organ_id must be provided when use_film=True")

        x = self.stem(x)

        # Layer 1
        x = self.layer1(x)
        if self.use_cbam:
            x = self.cbam1(x)
        if self.use_film:
            x = self.film1(x, organ_id)

        # Layer 2
        x = self.layer2(x)
        if self.use_cbam:
            x = self.cbam2(x)
        if self.use_film:
            x = self.film2(x, organ_id)

        # Layer 3
        x = self.layer3(x)
        if self.use_cbam:
            x = self.cbam3(x)
        if self.use_film:
            x = self.film3(x, organ_id)

        # Layer 4
        x = self.layer4(x)
        if self.use_cbam:
            x = self.cbam4(x)
        if self.use_film:
            x = self.film4(x, organ_id)

        # Final layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class MultiLoss(nn.Module):
    def __init__(
        self,
        device,
        wandb_logging=False,
        predict_bboxes=False,
        normalize_factor=256,
        focal_loss_gamma=2.0,
        focal_loss_alpha=0.25,
    ):
        super().__init__()
        self.wandb_logging = wandb_logging
        self.predict_bboxes = predict_bboxes
        self.device = device
        self.normalize_factor = normalize_factor
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha

    def forward(
        self,
        multi_cls_logits,
        bbox_logits,
        multi_cls_labels,
        unormalized_bbox_coords,
        organ_labels,
    ):
        multi_cls_mask = multi_cls_labels != -100
        dataset_names = list(organ_to_class_dict.keys())
        if multi_cls_mask.any():
            loss_multi = self._focal_loss(
                multi_cls_logits,
                multi_cls_labels,
                self.focal_loss_gamma,
                self.focal_loss_alpha,
                ignore_index=-100,
            )
        else:
            loss_multi = torch.zeros(1, device=self.device)

        single_loss = torch.zeros(1, device=self.device)
        for i, (logits, label, organ_label) in enumerate(zip(
            multi_cls_logits, multi_cls_labels, organ_labels
        )):
            # Inside your loop in cls_net.py
            if label != -100:
                dataset_name = dataset_names[organ_label]
                labels_set = multi_cls_labels_dict[dataset_name]
                label = label - labels_set[0]
                single_cls_out = logits[labels_set[0] : labels_set[-1] + 1]
                single_loss = single_loss + nn.functional.cross_entropy(
                    input=single_cls_out.unsqueeze(0),  # shape: 2
                    target=label.unsqueeze(0),  # shape 1
                    ignore_index=-100,
                )

        single_loss = single_loss / multi_cls_labels.shape[0]
        mask = unormalized_bbox_coords[:, 0, 0] != -100

        if self.predict_bboxes and mask.any():
            loss_bbox = nn.functional.smooth_l1_loss(
                input=bbox_logits[mask],
                target=unormalized_bbox_coords.squeeze(1)[mask] / self.normalize_factor,
                reduction="mean",
                beta=1.0,
            )
        else:
            loss_bbox = torch.zeros(1, device=self.device)

        if self.wandb_logging:
            wandb.log(
                {
                    f"loss_multi": loss_multi.item(),
                    f"loss_single": single_loss.item(),
                    f"loss_bbox": loss_bbox.item(),
                },
                commit=False,
            )
        return loss_multi + loss_bbox + single_loss

    def _focal_loss(
        self,
        inputs,
        targets,
        gamma=2.0,
        alpha=0.25,
        ignore_index=-100,
    ):
        """
        Focal Loss implementation.
        Based on the paper: "Focal Loss for Dense Object Detection" by Lin et al.
        Args:
            inputs (torch.Tensor): Logits of shape [N, C] where N is batch size and C is number of classes.
            targets (torch.Tensor): Ground truth labels of shape [N].
            gamma (float): Focusing parameter.
            alpha (float): Balancing parameter.
            ignore_index (int): Specifies a target value that is ignored.
        """
        ce_loss = nn.functional.cross_entropy(
            inputs,
            targets,
            reduction="none",
            ignore_index=ignore_index,
        )

        # Handle the case where the target is ignored
        if ignore_index != -100:
            valid_mask = targets != ignore_index
            if not valid_mask.any():
                return torch.zeros(1, device=self.device)
            ce_loss = ce_loss[valid_mask]
            inputs = inputs[valid_mask]
            targets = targets[valid_mask]

        if not inputs.numel():
            return torch.zeros(1, device=self.device)

        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
