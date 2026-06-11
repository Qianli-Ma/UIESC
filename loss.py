from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models

REPO_ROOT = Path(__file__).resolve().parents[4]
TRANSFUSE_LOSSES = REPO_ROOT / "experiments" / "models" / "transfuse-gan" / "src" / "losses"
if str(TRANSFUSE_LOSSES) not in sys.path:
    sys.path.insert(0, str(TRANSFUSE_LOSSES))

from sharpness_loss import MSSSIMLoss  # noqa: E402


class VGG19FeatureExtractor(nn.Module):
    def __init__(self, layer_ids: list[int]) -> None:
        super().__init__()
        weights = models.VGG19_Weights.IMAGENET1K_V1
        self.vgg = models.vgg19(weights=weights).features[: max(layer_ids) + 1].eval()
        self.layer_ids = set(layer_ids)
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = (x - self.mean) / self.std
        features: list[torch.Tensor] = []
        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            if idx in self.layer_ids:
                features.append(x)
        return features


class ContrastiveLoss(nn.Module):
    """UIESC contrastive ratio loss in VGG-19 latent feature space."""

    def __init__(
        self,
        layer_ids: list[int] | None = None,
        weights: list[float] | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        layer_ids = layer_ids or [1, 3, 5, 9, 13]
        weights = weights or [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1.0]
        if len(layer_ids) != len(weights):
            raise ValueError("contrastive layer_ids and weights must have the same length")
        self.extractor = VGG19FeatureExtractor(layer_ids)
        self.l1 = nn.L1Loss()
        self.eps = eps
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, source: torch.Tensor, enhanced: torch.Tensor, clear: torch.Tensor) -> torch.Tensor:
        clear_features = self.extractor(clear)
        enhanced_features = self.extractor(enhanced)
        source_features = self.extractor(source)

        loss = enhanced.new_tensor(0.0)
        for weight, positive, anchor, negative in zip(
            self.weights.to(enhanced.device),
            clear_features,
            enhanced_features,
            source_features,
            strict=True,
        ):
            d_pos = self.l1(anchor, positive.detach())
            d_neg = self.l1(anchor, negative.detach())
            loss = loss + weight * d_pos / (d_neg + self.eps)
        return loss


class UIESCLoss(nn.Module):
    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_msssim: float = 0.01,
        lambda_cl: float = 0.01,
        contrastive_layers: list[int] | None = None,
        contrastive_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_msssim = lambda_msssim
        self.lambda_cl = lambda_cl
        self.l1 = nn.L1Loss()
        self.msssim = MSSSIMLoss()
        self.contrastive = ContrastiveLoss(contrastive_layers, contrastive_weights)

    def forward(
        self,
        source: torch.Tensor,
        enhanced_lsan: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        l1 = self.l1(enhanced_lsan, target)
        msssim = self.msssim(enhanced_lsan, target)
        cl = self.contrastive(source, enhanced_lsan, target)
        total = self.lambda_l1 * l1 + self.lambda_msssim * msssim + self.lambda_cl * cl
        return total, {"l1": l1.detach(), "msssim": msssim.detach(), "cl": cl.detach()}
