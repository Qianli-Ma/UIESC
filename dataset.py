from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[0]


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


class UIEBDataset(Dataset):
    """Local copy of the paired UIEB loader used by this baseline."""

    def __init__(
        self,
        config: dict[str, Any],
        split: str,
        transform: Callable | None = None,
    ) -> None:
        data_cfg = config.get("data", {})
        self.input_dir = resolve_repo_path(data_cfg["uieb_input"])
        self.target_dir = resolve_repo_path(data_cfg["uieb_target"])
        self.uieb_split_dir = resolve_repo_path(data_cfg["uieb_split_dir"])
        self.split = split

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory '{self.input_dir}' does not exist.")
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Target directory '{self.target_dir}' does not exist.")
        if not self.uieb_split_dir.exists():
            raise FileNotFoundError(f"Split directory '{self.uieb_split_dir}' does not exist.")

        split_file = self.uieb_split_dir / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file '{split_file}' does not exist.")

        self.files = self._load_split(split_file)
        if not self.files:
            raise RuntimeError(f"No files found in split file '{split_file}'.")

        image_size = int(data_cfg.get("image_size", 256))
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def _load_split(self, split_file: Path) -> list[str]:
        with split_file.open("r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        filename = self.files[idx]
        input_path = self.input_dir / filename
        target_path = self.target_dir / filename

        if not input_path.exists():
            raise FileNotFoundError(f"Input file '{input_path}' does not exist.")
        if not target_path.exists():
            raise FileNotFoundError(f"Target file '{target_path}' does not exist.")

        with Image.open(input_path) as input_image:
            x = input_image.convert("RGB")
        with Image.open(target_path) as target_image:
            y = target_image.convert("RGB")

        return {"input": self.transform(x), "target": self.transform(y)}


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
