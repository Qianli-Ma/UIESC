from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[4]
TRANSFUSE_DATASETS = REPO_ROOT / "experiments" / "models" / "transfuse-gan" / "src" / "datasets"

if str(TRANSFUSE_DATASETS) not in sys.path:
    sys.path.insert(0, str(TRANSFUSE_DATASETS))

from uieb import UIEB as _TransFuseUIEB  # noqa: E402


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


class UIEBDataset(_TransFuseUIEB):
    """Thin UIESC wrapper around the existing TransFuse-GAN UIEB dataset."""

    def __init__(self, config: dict[str, Any], split: str) -> None:
        data_cfg = config.get("data", {})
        super().__init__(
            input_dir=resolve_repo_path(data_cfg["uieb_input"]),
            target_dir=resolve_repo_path(data_cfg["uieb_target"]),
            uieb_split_dir=resolve_repo_path(data_cfg["uieb_split_dir"]),
            split=split,
            image_size=int(data_cfg.get("image_size", 256)),
        )


def build_uieb_loader(
    config: dict[str, Any],
    split: str,
    shuffle: bool | None = None,
    drop_last: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    data_cfg = config.get("data", {})
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        UIEBDataset(config, split=split),
        batch_size=int(data_cfg.get("batch_size", 16)),
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 8)),
        drop_last=drop_last,
        pin_memory=pin_memory,
    )
