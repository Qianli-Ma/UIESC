from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataset import build_uieb_loader, resolve_repo_path
    from loss import UIESCLoss
    from model import UIESCModel
else:
    from .dataset import build_uieb_loader, resolve_repo_path
    from .loss import UIESCLoss
    from .model import UIESCModel


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(cfg: dict[str, Any]) -> UIESCModel:
    model_cfg = cfg.get("model", {})
    repeats = tuple(int(v) for v in model_cfg.get("lsa_repeats", [1, 1, 2]))
    return UIESCModel(
        base_channels=int(model_cfg.get("base_channels", 32)),
        lsa_repeats=repeats,
        she_bins=int(model_cfg.get("she_bins", 256)),
    )


def make_loss(cfg: dict[str, Any]) -> UIESCLoss:
    loss_cfg = cfg.get("loss", {})
    return UIESCLoss(
        lambda_l1=float(loss_cfg.get("lambda_l1", 1.0)),
        lambda_msssim=float(loss_cfg.get("lambda_msssim", 0.01)),
        lambda_cl=float(loss_cfg.get("lambda_cl", 0.01)),
        contrastive_layers=[int(x) for x in loss_cfg.get("contrastive_layers", [1, 3, 5, 9, 13])],
        contrastive_weights=[float(x) for x in loss_cfg.get("contrastive_weights", [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1.0])],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the UIESC baseline on UIEB.")
    parser.add_argument("--config", default="baselines/UIESC/config.yaml")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--inference-stats", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def validate_psnr(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    psnr_scores: list[float] = []
    l1_scores: list[float] = []
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        preds = model(inputs).clamp(0, 1)
        mse = torch.mean((preds - targets) ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
        psnr_scores.extend((10.0 * torch.log10(1.0 / mse)).detach().cpu().tolist())
        l1_scores.extend(torch.mean(torch.abs(preds - targets), dim=(1, 2, 3)).detach().cpu().tolist())
    model.train()
    return {"PSNR": float(np.mean(psnr_scores)), "L1": float(np.mean(l1_scores))}


@torch.no_grad()
def print_inference_stats(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    device: torch.device,
    use_amp: bool,
) -> None:
    stats_cfg = dict(cfg)
    stats_cfg.setdefault("data", {})
    stats_cfg["data"] = dict(stats_cfg["data"])
    stats_cfg["data"]["batch_size"] = 1
    stats_cfg["data"]["num_workers"] = 0
    loader = build_uieb_loader(stats_cfg, split="test", shuffle=False, drop_last=False, pin_memory=device.type == "cuda")

    total_images = 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    model.eval()
    autocast_enabled = False
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            autocast_enabled = torch.is_autocast_enabled()
            _ = model(inputs).clamp(0, 1)
        total_images += inputs.shape[0]

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    else:
        peak_vram = 0.0
    elapsed = time.perf_counter() - start
    params_m = sum(param.numel() for param in model.parameters()) / 1_000_000
    ms_per_image = (elapsed / max(total_images, 1)) * 1000.0

    print("--- Inference Stats ---")
    print(f"AMP enabled       : {autocast_enabled}")
    print(f"Parameter count   : {params_m:.2f} M")
    print(f"Total images      : {total_images}")
    print(f"Time per image    : {ms_per_image:.2f} ms")
    print(f"Peak VRAM         : {peak_vram:.3f} GB")
    print("Batch size        : 1")


def main() -> None:
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    cfg = load_config(config_path)

    if args.batch_size is not None:
        cfg.setdefault("data", {})["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg.setdefault("data", {})["num_workers"] = args.num_workers
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.lr is not None:
        cfg.setdefault("training", {})["lr"] = args.lr

    training_cfg = cfg.get("training", {})
    set_seed(int(training_cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    output_root = resolve_repo_path(cfg.get("output", {}).get("root", "baselines/UIESC/outputs"))
    checkpoint_dir = output_root / "checkpoints"
    tb_dir = output_root / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    model = make_model(cfg).to(device)

    start_epoch = 0
    best_val_psnr = float("-inf")
    if args.inference_stats:
        if args.resume is not None:
            state = torch.load(args.resume, map_location=device)
            model.load_state_dict(state["model"])
        print_inference_stats(model, cfg, device, use_amp)
        return

    criterion = make_loss(cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg.get("lr", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    scaler = GradScaler(enabled=use_amp)

    if args.resume is not None:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_val_psnr = float(state.get("best_val_psnr", best_val_psnr))

    train_loader = build_uieb_loader(cfg, split="train", shuffle=True, drop_last=True, pin_memory=use_amp)
    val_loader = build_uieb_loader(cfg, split="val", shuffle=False, drop_last=False, pin_memory=use_amp)
    writer = SummaryWriter(log_dir=str(tb_dir))

    epochs = int(training_cfg.get("epochs", 200))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 10))
    global_step = start_epoch * len(train_loader)

    print(f"Config: {config_path}")
    print(f"Device: {device} | AMP: {use_amp}")
    print(f"Output: {output_root}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_start = time.time()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")
        running = {"total": 0.0, "l1": 0.0, "msssim": 0.0, "cl": 0.0}
        seen_batches = 0

        for batch in progress:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            if device.type == "cuda" and not hasattr(model, "_vram_recorded"):
                torch.cuda.reset_peak_memory_stats()

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                enhanced_lsan = model.forward_lsan(inputs)
                loss, loss_parts = criterion(inputs, enhanced_lsan, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if device.type == "cuda" and not hasattr(model, "_vram_recorded"):
                peak = torch.cuda.max_memory_allocated() / 1024**3
                batch_size = int(cfg.get("data", {}).get("batch_size", 16))
                print(f"Peak training VRAM: {peak:.3f} GB  (AMP {'on' if use_amp else 'off'}, batch {batch_size})")
                model._vram_recorded = True

            running["total"] += loss.item()
            running["l1"] += loss_parts["l1"].item()
            running["msssim"] += loss_parts["msssim"].item()
            running["cl"] += loss_parts["cl"].item()
            seen_batches += 1
            global_step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}", l1=f"{loss_parts['l1'].item():.4f}")

        metrics = validate_psnr(model, val_loader, device)
        val_psnr = float(metrics["PSNR"])
        train_total = running["total"] / max(seen_batches, 1)
        train_l1 = running["l1"] / max(seen_batches, 1)
        train_msssim = running["msssim"] / max(seen_batches, 1)
        train_cl = running["cl"] / max(seen_batches, 1)
        writer.add_scalar("Loss/train_total", train_total, epoch)
        writer.add_scalar("Loss/train_l1", train_l1, epoch)
        writer.add_scalar("Loss/train_msssim", train_msssim, epoch)
        writer.add_scalar("Loss/train_cl", train_cl, epoch)
        writer.add_scalar("Metrics/val_psnr", val_psnr, epoch)

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_psnr": max(best_val_psnr, val_psnr),
            "config": cfg,
        }
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            torch.save(state, checkpoint_dir / f"epoch_{epoch + 1:04d}.pth")
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            state["best_val_psnr"] = best_val_psnr
            torch.save(state, checkpoint_dir / "best_generator.pth")

        elapsed = time.time() - epoch_start
        print(f"[Epoch {epoch + 1}/{epochs}] loss={train_total:.4f} | val_psnr={val_psnr:.4f} | best={best_val_psnr:.4f}")
        print(f"elapsed={elapsed:.1f}s")

    writer.close()


if __name__ == "__main__":
    main()
