"""
3D medical image segmentation training script (MONAI + PyTorch).

This module trains a 3D SegResNet model for binary segmentation using MONAI's
data pipeline and evaluation utilities. It supports AMP training, cosine LR
schedule with warmup, sliding-window inference for validation/test, checkpointing,
metric logging (Dice/F1), and saving qualitative example overlays and NIfTI
predictions.

Expected input:
- A CSV manifest file with columns: case_id, image_path, mask_path, split
  where split is one of: train / val / test.

Outputs (per run):
- config.json
- manifest_used.csv
- checkpoints/last.pt, checkpoints/best.pt, checkpoints/best_metric_model.pth
- plots/loss.png, plots/dice.png, plots/f1.png
- test_metrics.csv
- examples/*_overlay.png and *_pred.nii.gz
"""

import json
import time
import math
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

import nibabel as nib

from monai import data, transforms
from monai.data import CacheDataset
from monai.data.utils import list_data_collate
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import AsDiscrete, Compose
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference


# -------------------------- CONFIG -------------------------- #
@dataclass
class Config:
    """Configuration container for training, preprocessing, and evaluation.

    Attributes:
        manifest_rel: Relative path to the CSV manifest (from project root).
        seed: Random seed for reproducibility.
        val_from_train: If the manifest has no 'val' split, sample this fraction
            from the 'train' split to create validation.
        roi: 3D patch size used for training crops and sliding-window inference.
        batch_size: Training batch size (note: training uses random crops).
        num_crop_samples: Number of random crops sampled per volume per iteration.
        num_workers: DataLoader worker count.
        cache_rate: MONAI CacheDataset cache rate (0..1).
        sw_batch_size: Sliding window batch size for eval inference.
        sw_overlap: Sliding window overlap fraction.
        eval_use_sliding_window: If True, use sliding-window inference for val/test.
        pixdim: Target voxel spacing for images.
        use_orientation_ras: If True, reorient image/label to RAS.
        epochs: Number of training epochs.
        lr: Base learning rate.
        weight_decay: Weight decay for AdamW.
        amp: If True and CUDA is available, train with automatic mixed precision.
        grad_clip: Gradient norm clipping value (<=0 disables).
        dice_weight: Weight of Dice loss term.
        ce_weight: Weight of cross-entropy loss term (0 disables CE).
        pos_class_weight: Positive class weight for cross-entropy (class 1).
        use_cosine: If True, use cosine annealing LR schedule after warmup.
        warmup_epochs: Linear warmup epochs for LR (only if use_cosine is True).
        eval_max_batches: Max number of evaluation batches (large number = no cap).
        run_name: Optional run name; if empty, timestamp will be used.
        examples_n: Number of test examples to export as overlays and NIfTI preds.
        threshold: Reserved for probability thresholding (not used in argmax pipeline).
    """

    manifest_rel: str = "data/manifest.csv"
    seed: int = 42

    # if no val -> sample from train
    val_from_train: float = 0.10

    # roi / loaders
    roi: Tuple[int, int, int] = (96, 96, 96)
    batch_size: int = 1
    num_crop_samples: int = 4
    num_workers: int = 2
    cache_rate: float = 0.2

    # sliding window for val/test (IMPORTANT for SegResNet)
    sw_batch_size: int = 1
    sw_overlap: float = 0.25
    eval_use_sliding_window: bool = True  # key fix for full-volume eval

    # preprocess
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    use_orientation_ras: bool = True

    # training
    epochs: int = 10
    lr: float = 2e-4
    weight_decay: float = 1e-5
    amp: bool = True
    grad_clip: float = 1.0

    # loss
    dice_weight: float = 1.0
    ce_weight: float = 0.5
    pos_class_weight: float = 5.0

    # scheduler
    use_cosine: bool = True
    warmup_epochs: int = 1

    # eval
    eval_max_batches: int = 999999

    # outputs
    run_name: str = ""
    examples_n: int = 3
    threshold: float = 0.5


# -------------------------- PATHS / UTILS -------------------------- #
def project_root() -> Path:
    """Return the project root directory.

    The project root is assumed to be two levels above this file:
    Path(__file__).resolve().parents[2].

    Returns:
        Absolute Path to the project root.
    """
    return Path(__file__).resolve().parents[2]


def set_seed(seed: int):
    """Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Seed value for deterministic-ish behavior.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_run_dir(model_dir: Path, run_name: str) -> Path:
    """Create and return a run directory with standard subfolders.

    If run_name is empty, a timestamp-based name is used.

    Structure:
        runs/<run_name>/
            checkpoints/
            plots/
            examples/

    Args:
        model_dir: Directory where the training script lives.
        run_name: Optional run directory name.

    Returns:
        Path to the created run directory.
    """
    if not run_name:
        run_name = time.strftime("%Y%m%d_%H%M%S")
    run_dir = model_dir / "runs" / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "examples").mkdir(parents=True, exist_ok=True)
    return run_dir


def gpu_check(run_dir: Path):
    """Collect and write basic CUDA/GPU information.

    Writes a text file `gpu_info.txt` into run_dir and prints the same info.

    Args:
        run_dir: Run directory where gpu_info.txt will be stored.
    """
    lines = []
    lines.append(f"torch.__version__ = {torch.__version__}")
    lines.append(f"cuda available     = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        lines.append(f"cuda version       = {torch.version.cuda}")
        lines.append(f"device count       = {torch.cuda.device_count()}")
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        lines.append(f"current device     = {idx} ({props.name})")
        lines.append(f"total memory (GB)  = {props.total_memory / (1024**3):.2f}")
    else:
        lines.append("GPU not detected: training will run on CPU (very slow).")

    text = "\n".join(lines)
    print("\n=== GPU CHECK ===")
    print(text)
    print("=================\n")
    (run_dir / "gpu_info.txt").write_text(text, encoding="utf-8")


def save_history_csv(path: Path, rows: List[Dict]):
    """Save training history as a CSV.

    Args:
        path: Output CSV path.
        rows: List of dicts with consistent keys (columns).
    """
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_history(history_csv: Path, out_dir: Path):
    """Plot learning curves (loss, dice, f1) from history.csv.

    Args:
        history_csv: Path to the history CSV file.
        out_dir: Directory where plots will be saved as PNG files.
    """
    df = pd.read_csv(history_csv)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["train_dice"], label="train_dice")
    plt.plot(df["epoch"], df["val_dice"], label="val_dice")
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "dice.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["train_f1"], label="train_f1")
    plt.plot(df["epoch"], df["val_f1"], label="val_f1")
    plt.xlabel("epoch")
    plt.ylabel("f1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "f1.png", dpi=150)
    plt.close()


def resolve_path(p: str) -> str:
    """Resolve a path string to an absolute path.

    If p is already absolute, it is returned unchanged.
    Otherwise, it is resolved relative to the project root.

    Args:
        p: Input path string.

    Returns:
        Absolute path string.
    """
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((project_root() / pp).resolve())


def load_manifest(cfg: Config) -> pd.DataFrame:
    """Load and validate the dataset manifest.

    The manifest must include columns: case_id, image_path, mask_path.
    Optional column: split.

    Paths in the manifest are converted to absolute paths.

    Args:
        cfg: Configuration with manifest_rel.

    Returns:
        Pandas DataFrame containing the manifest.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        ValueError: If required columns are missing.
    """
    p = project_root() / cfg.manifest_rel
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")

    df = pd.read_csv(p)
    need = ["case_id", "image_path", "mask_path"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Manifest must contain {need}. Found: {df.columns.tolist()}")

    df["case_id"] = df["case_id"].astype(str).str.strip()
    df["image_path"] = df["image_path"].astype(str).map(resolve_path)
    df["mask_path"] = df["mask_path"].astype(str).map(resolve_path)

    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.lower().str.strip()

    return df


def make_splits(cfg: Config, df: pd.DataFrame):
    """Create train/val/test splits based on the manifest.

    Expects a 'split' column. If there is no 'val' split, a portion of the train
    rows is sampled and reassigned to 'val'.

    Args:
        cfg: Configuration with seed and val_from_train.
        df: Manifest DataFrame with 'split' column.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If 'split' column is missing.
    """
    if "split" not in df.columns:
        raise ValueError("Expected 'split' column in data/manifest.csv (you said you already split it).")

    if "val" not in set(df["split"].unique()):
        tr = df[df["split"] == "train"].copy()
        val = tr.sample(frac=cfg.val_from_train, random_state=cfg.seed)
        df = df.copy()
        df.loc[val.index, "split"] = "val"

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df


def make_items(df: pd.DataFrame) -> List[Dict]:
    """Convert a manifest DataFrame to MONAI-style item dicts.

    Args:
        df: DataFrame with columns image_path, mask_path, case_id.

    Returns:
        List of dicts with keys: image, label, case_id.
    """
    return [{"image": r["image_path"], "label": r["mask_path"], "case_id": r["case_id"]} for _, r in df.iterrows()]


def safe_nan_to_num_np(x):
    """Replace NaN/Inf values with zeros in a NumPy array.

    Args:
        x: NumPy array.

    Returns:
        Array with NaN, +Inf, -Inf replaced by 0.0.
    """
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


# -------------------------- TRANSFORMS -------------------------- #
def build_transforms(cfg: Config):
    """Build MONAI transforms for training and validation/testing.

    Training pipeline:
        - Load image/label
        - Optional reorientation to RAS
        - Resample image spacing, match label to image grid
        - Pad to ROI
        - Normalize intensity on nonzero region
        - Replace NaN/Inf
        - EnsureTyped
        - Random crop by pos/neg labels (num_samples)
        - Augmentations (rotation, flips, zoom, affine, contrast/intensity/noise)

    Validation/testing pipeline:
        Same as base preprocessing without random crops and augmentation.

    Args:
        cfg: Configuration.

    Returns:
        Tuple of (train_transform, val_transform).
    """
    base = [
        transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
    ]
    if cfg.use_orientation_ras:
        # labels=None avoids future warnings and uses meta-tensor when available
        base += [transforms.Orientationd(keys=["image", "label"], axcodes="RAS", labels=None)]

    base += [
        transforms.Spacingd(keys=["image"], pixdim=cfg.pixdim, mode=("bilinear",)),
        transforms.ResampleToMatchd(keys=["label"], key_dst="image", mode="nearest"),
        transforms.SpatialPadd(keys=["image", "label"], spatial_size=cfg.roi),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        transforms.Lambdad(keys=["image"], func=safe_nan_to_num_np),
        transforms.EnsureTyped(keys=["image", "label"]),
    ]

    train_crop = [
        transforms.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=cfg.roi,
            num_samples=cfg.num_crop_samples,
            image_key="image",
            allow_smaller=True,
        )
    ]

    aug = [
        transforms.RandRotated(
            keys=["image", "label"],
            range_x=np.pi / 6,
            range_y=np.pi / 6,
            range_z=np.pi / 6,
            prob=0.3,
            mode=("bilinear", "nearest"),
            align_corners=True,
        ),
        transforms.RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.3),
        transforms.RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.3),
        transforms.RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.3),
        transforms.RandZoomd(
            keys=["image", "label"],
            min_zoom=0.9,
            max_zoom=1.1,
            mode=("trilinear", "nearest"),
            align_corners=(True, None),
            prob=0.3,
        ),
        transforms.RandAffined(
            keys=["image", "label"],
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.1, 0.1, 0.1),
            translate_range=(5, 5, 5),
            prob=0.3,
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),
        transforms.RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.5), prob=0.3),
        transforms.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
        transforms.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
        transforms.RandGaussianNoised(keys=["image"], std=0.05, prob=0.3),
    ]

    train_tf = transforms.Compose(base + train_crop + aug)
    val_tf = transforms.Compose(base)
    return train_tf, val_tf


# -------------------------- METRICS / VIZ -------------------------- #
def binary_f1_from_logits(logits_2ch: torch.Tensor, labels_1ch: torch.Tensor, eps=1e-8) -> float:
    """Compute binary F1 score from 2-channel logits and 1-channel label tensor.

    Prediction is obtained via argmax over channel dimension.
    Ground truth is treated as positive if label > 0.

    Args:
        logits_2ch: Tensor of shape [B, 2, ...] (raw logits).
        labels_1ch: Tensor of shape [B, 1, ...] (original label tensor).
        eps: Numerical stability constant.

    Returns:
        Scalar F1 score as float.
    """
    pred = torch.argmax(logits_2ch, dim=1).float()
    gt = (labels_1ch[:, 0] > 0).float()
    tp = torch.sum(pred * gt)
    fp = torch.sum(pred * (1 - gt))
    fn = torch.sum((1 - pred) * gt)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return float(f1.item())


def three_view_overlay_png(out_png: Path, img3d: np.ndarray, gt3d: np.ndarray, pr3d: np.ndarray, title: str):
    """Save a 3-panel (sagittal/coronal/axial) overlay visualization as PNG.

    The slice indices are chosen by finding the maximum projected ground-truth mask
    area along each axis.

    Args:
        out_png: Output PNG path.
        img3d: 3D image array [X, Y, Z].
        gt3d: 3D ground-truth mask array [X, Y, Z].
        pr3d: 3D predicted mask array [X, Y, Z].
        title: Figure title prefix.
    """
    lo, hi = np.percentile(img3d, (1, 99))
    imgv = np.clip(img3d, lo, hi)

    m = (gt3d > 0).astype(np.uint8)
    x = int(np.argmax(m.sum(axis=(1, 2))))
    y = int(np.argmax(m.sum(axis=(0, 2))))
    z = int(np.argmax(m.sum(axis=(0, 1))))

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(imgv[x, :, :].T, cmap="gray", origin="lower")
    plt.contour((gt3d[x, :, :] > 0).T, levels=[0.5], origin="lower")
    plt.contour((pr3d[x, :, :] > 0).T, levels=[0.5], origin="lower")
    plt.title(f"{title}\nSagittal x={x}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(imgv[:, y, :].T, cmap="gray", origin="lower")
    plt.contour((gt3d[:, y, :] > 0).T, levels=[0.5], origin="lower")
    plt.contour((pr3d[:, y, :] > 0).T, levels=[0.5], origin="lower")
    plt.title(f"Coronal y={y}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(imgv[:, :, z].T, cmap="gray", origin="lower")
    plt.contour((gt3d[:, :, z] > 0).T, levels=[0.5], origin="lower")
    plt.contour((pr3d[:, :, z] > 0).T, levels=[0.5], origin="lower")
    plt.title(f"Axial z={z}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# -------------------------- MODEL -------------------------- #
def build_model(cfg: Config, device: torch.device) -> nn.Module:
    """Create and move the MONAI SegResNet model to the target device.

    Args:
        cfg: Configuration (currently not used for model params, kept for extensibility).
        device: Target torch device.

    Returns:
        SegResNet model on the given device.
    """
    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=32,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.1,
    ).to(device)
    print("Using SegResNet (MONAI).")
    return model


def build_scheduler(cfg: Config, optimizer):
    """Create a cosine annealing learning rate scheduler (optional).

    Warmup is handled externally in the training loop by directly setting LR.

    Args:
        cfg: Configuration with scheduler settings.
        optimizer: Torch optimizer.

    Returns:
        CosineAnnealingLR scheduler or None if disabled.
    """
    if not cfg.use_cosine:
        return None
    t_max = max(1, cfg.epochs - cfg.warmup_epochs)
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=cfg.lr * 0.05)


def current_lr(optimizer):
    """Get the current learning rate from the optimizer.

    Args:
        optimizer: Torch optimizer.

    Returns:
        Current LR (float).
    """
    return optimizer.param_groups[0]["lr"]


def set_lr(optimizer, lr):
    """Set learning rate for all parameter groups.

    Args:
        optimizer: Torch optimizer.
        lr: New learning rate value.
    """
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def forward_for_eval(cfg: Config, model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Forward pass for validation/test, optionally using sliding-window inference.

    SegResNet can fail on full volumes due to skip-connection size mismatches
    when spatial sizes are not compatible. Sliding-window inference evaluates the
    model on patches of size cfg.roi and stitches them back.

    Args:
        cfg: Configuration with eval sliding-window settings.
        model: Segmentation model.
        x: Input tensor [B, C, X, Y, Z].
        device: Torch device for inference.

    Returns:
        Logits tensor [B, 2, X, Y, Z] (stitched if sliding window is used).
    """
    if not cfg.eval_use_sliding_window:
        return model(x)

    return sliding_window_inference(
        inputs=x,
        roi_size=cfg.roi,
        sw_batch_size=cfg.sw_batch_size,
        predictor=model,
        overlap=cfg.sw_overlap,
        mode="gaussian",
        sw_device=device,
        device=device,
        progress=False,
    )


# -------------------------- TRAIN / EVAL -------------------------- #
def train_one_epoch(
    model,
    loader,
    optimizer,
    dice_loss,
    ce_loss,
    cfg,
    device,
    scaler,
    post_pred,
    post_label,
    dice_metric,
):
    """Run one training epoch.

    Training is performed on random crops (RandCropByPosNegLabeld) so the model
    always sees ROI-sized patches.

    Metrics:
        - Dice: computed using MONAI DiceMetric on one-hot predictions/labels.
        - F1: computed on argmax predictions vs binary ground truth.

    Args:
        model: Segmentation model.
        loader: Training DataLoader.
        optimizer: Optimizer instance.
        dice_loss: MONAI DiceLoss instance.
        ce_loss: CrossEntropyLoss instance.
        cfg: Configuration.
        device: Torch device.
        scaler: GradScaler for AMP (can be disabled).
        post_pred: Post-processing transform for predictions (discretization).
        post_label: Post-processing transform for labels (one-hot).
        dice_metric: MONAI DiceMetric instance.

    Returns:
        Tuple: (epoch_loss, mean_dice, mean_f1)
    """
    model.train()
    epoch_loss = 0.0
    n_steps = 0
    train_dice_vals = []
    train_f1_vals = []

    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        y_bin = (y > 0).long()

        optimizer.zero_grad(set_to_none=True)

        use_amp = cfg.amp and torch.cuda.is_available()
        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(x)  # training always uses crops
            loss = cfg.dice_weight * dice_loss(logits, y_bin)
            if cfg.ce_weight > 0:
                loss = loss + cfg.ce_weight * ce_loss(logits, y_bin[:, 0])

        if not torch.isfinite(loss):
            print("[WARN] Non-finite loss -> skipping batch")
            continue

        if use_amp:
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        epoch_loss += float(loss.item())
        n_steps += 1

        with torch.no_grad():
            preds = [post_pred(i) for i in decollate_batch(logits)]
            labs = [post_label(i) for i in decollate_batch(y_bin)]
            dice_metric(y_pred=preds, y=labs)
            d = dice_metric.aggregate().item()
            dice_metric.reset()
            train_dice_vals.append(d)
            train_f1_vals.append(binary_f1_from_logits(logits, y))

    epoch_loss = epoch_loss / max(1, n_steps)
    train_dice = float(np.mean(train_dice_vals)) if train_dice_vals else 0.0
    train_f1 = float(np.mean(train_f1_vals)) if train_f1_vals else 0.0
    return epoch_loss, train_dice, train_f1


@torch.no_grad()
def evaluate(model, loader, dice_loss, ce_loss, cfg, device, post_pred, post_label, dice_metric):
    """Evaluate the model on a validation/test DataLoader.

    Uses sliding-window inference if cfg.eval_use_sliding_window is enabled.

    Args:
        model: Segmentation model.
        loader: Validation/Test DataLoader.
        dice_loss: MONAI DiceLoss instance.
        ce_loss: CrossEntropyLoss instance.
        cfg: Configuration.
        device: Torch device.
        post_pred: Post-processing transform for predictions (discretization).
        post_label: Post-processing transform for labels (one-hot).
        dice_metric: MONAI DiceMetric instance.

    Returns:
        Tuple: (mean_loss, dice, mean_f1)
    """
    model.eval()
    val_losses = []
    val_f1_vals = []
    batches = 0

    use_amp = cfg.amp and torch.cuda.is_available()
    for batch in tqdm(loader, desc="val", leave=False):
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        y_bin = (y > 0).long()

        with autocast(device_type="cuda", enabled=use_amp):
            logits = forward_for_eval(cfg, model, x, device)

            loss = cfg.dice_weight * dice_loss(logits, y_bin)
            if cfg.ce_weight > 0:
                loss = loss + cfg.ce_weight * ce_loss(logits, y_bin[:, 0])

        val_losses.append(float(loss.item()))

        preds = [post_pred(i) for i in decollate_batch(logits)]
        labs = [post_label(i) for i in decollate_batch(y_bin)]
        dice_metric(y_pred=preds, y=labs)

        val_f1_vals.append(binary_f1_from_logits(logits, y))

        batches += 1
        if batches >= cfg.eval_max_batches:
            break

    val_loss = float(np.mean(val_losses)) if val_losses else 0.0
    val_dice = dice_metric.aggregate().item() if batches > 0 else 0.0
    dice_metric.reset()
    val_f1 = float(np.mean(val_f1_vals)) if val_f1_vals else 0.0
    return val_loss, float(val_dice), val_f1


@torch.no_grad()
def save_examples(model, df_test: pd.DataFrame, run_dir: Path, cfg: Config, device):
    """Save qualitative prediction examples on the test set.

    For a small subset of test cases:
        - Run preprocessing (val transforms)
        - Run inference (sliding-window if enabled)
        - Save 3-view PNG overlays (image + GT contour + prediction contour)
        - Save predicted segmentation as NIfTI (.nii.gz) using the GT affine

    Args:
        model: Trained segmentation model.
        df_test: Test split DataFrame.
        run_dir: Run directory (examples will be saved to run_dir/examples).
        cfg: Configuration with examples_n.
        device: Torch device.
    """
    model.eval()
    _, val_tf = build_transforms(cfg)
    ex_dir = run_dir / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)

    pick = df_test.sample(n=min(cfg.examples_n, len(df_test)), random_state=cfg.seed).reset_index(drop=True)
    use_amp = cfg.amp and torch.cuda.is_available()

    for i in range(len(pick)):
        case_id = str(pick.loc[i, "case_id"])
        img_path = str(pick.loc[i, "image_path"])
        msk_path = str(pick.loc[i, "mask_path"])

        sample = val_tf({"image": img_path, "label": msk_path})
        x = sample["image"].unsqueeze(0).to(device)
        y = sample["label"].unsqueeze(0).to(device)

        with autocast(device_type="cuda", enabled=use_amp):
            logits = forward_for_eval(cfg, model, x, device)

        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
        gt = (y[0, 0].cpu().numpy() > 0).astype(np.uint8)
        img = x[0, 0].cpu().numpy()

        out_png = ex_dir / f"case_{case_id}_overlay.png"
        three_view_overlay_png(out_png, img, gt, pred, title=f"case {case_id}")

        affine = nib.load(msk_path).affine
        out_nii = nib.Nifti1Image(pred, affine=affine)
        out_nii.set_data_dtype(np.uint8)
        out_pred = ex_dir / f"case_{case_id}_pred.nii.gz"
        nib.save(out_nii, str(out_pred))

        print(f"[example] saved: {out_png.name}, {out_pred.name}")


def main():
    """Entry point for training, evaluation, and artifact export.

    Workflow:
        1. Create config and set seeds
        2. Create run directory and write config
        3. Load manifest and create splits
        4. Build transforms, datasets, loaders
        5. Build model, losses, optimizer, scheduler
        6. Train for cfg.epochs with warmup + cosine schedule
        7. Save checkpoints and plots each epoch
        8. Evaluate on test split and save metrics
        9. Save qualitative examples and NIfTI predictions
    """
    cfg = Config()
    set_seed(cfg.seed)

    torch.backends.cudnn.benchmark = True

    model_dir = Path(__file__).resolve().parent
    run_dir = ensure_run_dir(model_dir, cfg.run_name)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_check(run_dir)

    df = load_manifest(cfg)
    train_df, val_df, test_df = make_splits(cfg, df)

    print(f"Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")
    pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ]
    ).to_csv(run_dir / "manifest_used.csv", index=False)

    train_items = make_items(train_df)
    val_items = make_items(val_df)
    test_items = make_items(test_df)

    train_tf, val_tf = build_transforms(cfg)

    train_ds = CacheDataset(train_items, transform=train_tf, cache_rate=cfg.cache_rate, num_workers=0)
    val_ds = CacheDataset(val_items, transform=val_tf, cache_rate=cfg.cache_rate, num_workers=0)
    test_ds = CacheDataset(test_items, transform=val_tf, cache_rate=cfg.cache_rate, num_workers=0)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, cfg.num_workers // 2),
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, cfg.num_workers // 2),
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(cfg, device)

    dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
    ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, cfg.pos_class_weight], device=device))

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler("cuda", enabled=(cfg.amp and torch.cuda.is_available()))
    cosine = build_scheduler(cfg, optimizer)

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    history = []
    history_csv = run_dir / "history.csv"
    best_dice = -1.0

    for epoch in range(1, cfg.epochs + 1):
        # Linear warmup for LR (only when cosine schedule is enabled)
        if cfg.use_cosine and epoch <= cfg.warmup_epochs:
            lr = cfg.lr * (epoch / max(1, cfg.warmup_epochs))
            set_lr(optimizer, lr)

        t0 = time.time()

        tr_loss, tr_dice, tr_f1 = train_one_epoch(
            model,
            train_loader,
            optimizer,
            dice_loss,
            ce_loss,
            cfg,
            device,
            scaler,
            post_pred,
            post_label,
            dice_metric,
        )

        val_loss, val_dice, val_f1 = evaluate(
            model, val_loader, dice_loss, ce_loss, cfg, device, post_pred, post_label, dice_metric
        )

        if cosine is not None and epoch > cfg.warmup_epochs:
            cosine.step()

        dt = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "train_dice": tr_dice,
            "val_dice": val_dice,
            "train_f1": tr_f1,
            "val_f1": val_f1,
            "lr": current_lr(optimizer),
            "sec": dt,
        }
        history.append(row)
        save_history_csv(history_csv, history)
        plot_history(history_csv, run_dir / "plots")

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} | "
            f"train_dice={tr_dice:.4f} val_dice={val_dice:.4f} | "
            f"train_f1={tr_f1:.4f} val_f1={val_f1:.4f} | "
            f"lr={current_lr(optimizer):.2e} time={dt:.1f}s"
        )

        last_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cfg": asdict(cfg),
            "history": history,
        }
        torch.save(last_ckpt, run_dir / "checkpoints" / "last.pt")

        if math.isfinite(val_dice) and val_dice > best_dice:
            best_dice = val_dice
            torch.save(last_ckpt, run_dir / "checkpoints" / "best.pt")
            torch.save(model.state_dict(), run_dir / "checkpoints" / "best_metric_model.pth")
            print(f"  -> new best dice={best_dice:.4f} saved best.pt + best_metric_model.pth")

    print("\n=== TEST ===")
    test_loss, test_dice, test_f1 = evaluate(
        model, test_loader, dice_loss, ce_loss, cfg, device, post_pred, post_label, dice_metric
    )
    print(f"TEST loss={test_loss:.4f} dice={test_dice:.4f} f1={test_f1:.4f}")
    pd.DataFrame([{"test_loss": test_loss, "test_dice": test_dice, "test_f1": test_f1}]).to_csv(
        run_dir / "test_metrics.csv", index=False
    )

    print("\n=== Examples ===")
    save_examples(model, test_df, run_dir, cfg, device)

    print("\nDONE. Run dir:", run_dir.resolve())


if __name__ == "__main__":
    main()
