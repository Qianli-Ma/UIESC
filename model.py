from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CrissCrossSpatialAttention(nn.Module):
    """Row and column self-attention following the criss-cross attention idea."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        reduced = max(1, channels // 8)
        self.query = nn.Conv2d(channels, reduced, 1, bias=False)
        self.key = nn.Conv2d(channels, reduced, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q_row = q.permute(0, 2, 3, 1).reshape(b * h, w, -1)
        k_row = k.permute(0, 2, 1, 3).reshape(b * h, -1, w)
        v_row = v.permute(0, 2, 3, 1).reshape(b * h, w, c)
        row_attn = torch.softmax(torch.bmm(q_row, k_row), dim=-1)
        row_out = torch.bmm(row_attn, v_row).reshape(b, h, w, c).permute(0, 3, 1, 2)

        q_col = q.permute(0, 3, 2, 1).reshape(b * w, h, -1)
        k_col = k.permute(0, 3, 1, 2).reshape(b * w, -1, h)
        v_col = v.permute(0, 3, 2, 1).reshape(b * w, h, c)
        col_attn = torch.softmax(torch.bmm(q_col, k_col), dim=-1)
        col_out = torch.bmm(col_attn, v_col).reshape(b, w, h, c).permute(0, 3, 2, 1)

        return x + self.gamma * 0.5 * (row_out + col_out)


class ChannelAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = x.reshape(b, c, h * w)
        energy = torch.bmm(flat, flat.transpose(1, 2))
        attention = torch.softmax(energy, dim=-1)
        out = torch.bmm(attention, flat).reshape(b, c, h, w)
        return x + self.beta * out


class LSAModule(nn.Module):
    """Lightweight self-attention module with parallel spatial/channel branches."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial = CrissCrossSpatialAttention(channels)
        self.channel = ChannelAttention()
        self.aggregate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial(x)
        channel = self.channel(x)
        return x + self.aggregate(torch.cat([spatial, channel], dim=1))


class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.lam = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, encoder: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
        if encoder.shape[-2:] != decoder.shape[-2:]:
            decoder = F.interpolate(decoder, size=encoder.shape[-2:], mode="bilinear", align_corners=False)
        enc_weight = torch.sigmoid(self.lam)
        dec_weight = torch.sigmoid(1.0 - self.lam)
        return enc_weight * encoder + dec_weight * decoder


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.refine = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.up(x))


class LSAN(nn.Module):
    """Autoencoder-style Lightweight Self-Attention Network."""

    def __init__(self, base_channels: int = 32, lsa_repeats: tuple[int, int, int] = (1, 1, 2)) -> None:
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.stem = ConvBlock(3, c1)
        self.down1 = ConvBlock(c1, c2, stride=2)
        self.lsa1 = nn.Sequential(*[LSAModule(c2) for _ in range(lsa_repeats[0])])
        self.down2 = ConvBlock(c2, c3, stride=2)
        self.lsa2 = nn.Sequential(*[LSAModule(c3) for _ in range(lsa_repeats[1])])
        self.bottleneck = nn.Sequential(*[LSAModule(c3) for _ in range(lsa_repeats[2])])

        self.up1 = UpBlock(c3, c2)
        self.aff1 = AdaptiveFeatureFusion(c2)
        self.refine1 = ConvBlock(c2, c2)
        self.up2 = UpBlock(c2, c1)
        self.aff2 = AdaptiveFeatureFusion(c1)
        self.refine2 = ConvBlock(c1, c1)
        self.out = nn.Conv2d(c1, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.stem(x)
        e1 = self.lsa1(self.down1(e0))
        e2 = self.lsa2(self.down2(e1))
        z = self.bottleneck(e2)
        d1 = self.refine1(self.aff1(e1, self.up1(z)))
        d0 = self.refine2(self.aff2(e0, self.up2(d1)))
        residual = torch.tanh(self.out(d0))
        return (x + residual).clamp(0, 1)


class SmoothedHistogramEqualization(nn.Module):
    def __init__(self, bins: int = 256) -> None:
        super().__init__()
        self.bins = bins

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().clamp(0, 1)
        out = torch.empty_like(x)
        for b in range(x.shape[0]):
            for c in range(x.shape[1]):
                channel = x[b, c]
                if torch.isclose(channel.max(), channel.min()):
                    out[b, c] = channel
                    continue
                idx = torch.clamp((channel * (self.bins - 1)).round().long(), 0, self.bins - 1)
                hist = torch.bincount(idx.flatten(), minlength=self.bins).to(dtype=x.dtype, device=x.device)
                smoothed = torch.log1p(hist)
                total = smoothed.sum()
                if total <= 0:
                    out[b, c] = channel
                    continue
                lut = torch.cumsum(smoothed / total, dim=0).clamp(0, 1)
                out[b, c] = lut[idx]
        return out


class UIESCModel(nn.Module):
    """Full two-stage UIESC forward: LSAN followed by gradient-free SHE."""

    def __init__(
        self,
        base_channels: int = 32,
        lsa_repeats: tuple[int, int, int] = (1, 1, 2),
        she_bins: int = 256,
    ) -> None:
        super().__init__()
        self.lsan = LSAN(base_channels=base_channels, lsa_repeats=lsa_repeats)
        self.she = SmoothedHistogramEqualization(bins=she_bins)

    def forward_lsan(self, x: torch.Tensor) -> torch.Tensor:
        return self.lsan(x).clamp(0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stage1 = self.forward_lsan(x)
        return self.she(stage1)


def _count_parameters(model: nn.Module) -> float:
    return sum(param.numel() for param in model.parameters()) / 1_000_000


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UIESCModel().to(device).eval()
    sample = torch.rand(1, 3, 256, 256, device=device)
    with torch.no_grad():
        output = model(sample)
    print(f"output shape: {tuple(output.shape)}")
    print(f"parameter count: {_count_parameters(model):.3f} M")
