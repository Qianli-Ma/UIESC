from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def _gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    g = g / g.sum()
    kernel_2d = g[:, None] @ g[None, :]
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def _ssim_per_scale(
    pred: torch.Tensor,
    target: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    c1: float,
    c2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    channels = pred.shape[1]
    pad = window_size // 2

    mu_p = F.conv2d(pred, window, padding=pad, groups=channels)
    mu_t = F.conv2d(target, window, padding=pad, groups=channels)

    mu_p_sq = mu_p**2
    mu_t_sq = mu_t**2
    mu_pt = mu_p * mu_t

    sigma_p_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_p_sq
    sigma_t_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu_t_sq
    sigma_pt = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_pt

    cs_map = (2.0 * sigma_pt + c2) / (sigma_p_sq + sigma_t_sq + c2)
    ssim_map = ((2.0 * mu_pt + c1) / (mu_p_sq + mu_t_sq + c1)) * cs_map
    return ssim_map.mean(), cs_map.mean()


class MSSSIMLoss(nn.Module):
    """Local copy of the TransFuse-GAN MS-SSIM loss."""

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channels: int = 3,
        data_range: float = 1.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.register_buffer(
            "weights",
            torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=torch.float32),
        )
        self.c1 = (0.01 * data_range) ** 2
        self.c2 = (0.03 * data_range) ** 2
        self.register_buffer("window", _gaussian_window(window_size, sigma, channels))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # MS-SSIM uses squared terms and gaussian convolutions that overflow in
        # fp16. Force fp32 regardless of the surrounding autocast context.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            pred = pred.float()
            target = target.float()
            window = self.window.float()
            weights = self.weights.float()

            cs_values: list[torch.Tensor] = []
            ssim_value = pred.new_tensor(0.0)
            for scale in range(weights.numel()):
                ssim_value, cs_value = _ssim_per_scale(
                    pred, target, window, self.window_size, self.c1, self.c2
                )
                cs_values.append(cs_value.clamp(min=0.0))
                if scale < weights.numel() - 1:
                    pred = F.avg_pool2d(pred, kernel_size=2)
                    target = F.avg_pool2d(target, kernel_size=2)

            ssim_value = ssim_value.clamp(min=0.0)
            cs_stack = torch.stack(cs_values)
            ms_ssim = torch.prod(cs_stack**weights) * (ssim_value ** weights[-1])
            return 1.0 - ms_ssim


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
        # VGG-19 feature norms can overflow in fp16. Run extraction in fp32.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            clear_features = self.extractor(clear.float())
            enhanced_features = self.extractor(enhanced.float())
            source_features = self.extractor(source.float())

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
