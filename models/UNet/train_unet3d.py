# models/UNet/train_unet3d_like_kaggle.py
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

from sklearn.model_selection import train_test_split

import nibabel as nib

from monai import data, transforms
from monai.data import CacheDataset
from monai.data.utils import list_data_collate
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet, BasicUNet
from monai.transforms import AsDiscrete, Compose
from monai.data import decollate_batch


# -------------------------- CONFIG -------------------------- #
@dataclass
class Config:
    manifest_rel: str = "data/manifest.csv"
    seed: int = 42

    # если split нет
    test_size: float = 0.20
    val_from_train: float = 0.10

    # roi & loaders
    roi: Tuple[int, int, int] = (96, 96, 96)
    batch_size: int = 1
    num_crop_samples: int = 4
    num_workers: int = 2
    cache_rate: float = 0.2

    # preprocess
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    use_orientation_ras: bool = True

    # training
    epochs: int = 3
    lr: float = 1e-4
    use_basic_unet: bool = True
    amp: bool = False          # как в ноутбуке
    grad_clip: float = 1.0

    # loss
    dice_weight: float = 1.0
    ce_weight: float = 0.0     # можно поставить 0.5 для стабильности
    pos_class_weight: float = 5.0

    # eval speed
    eval_max_batches: int = 999999

    # outputs
    run_name: str = ""
    examples_n: int = 3
    threshold: float = 0.5


# -------------------------- PATHS / UTILS -------------------------- #
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_run_dir(unet_dir: Path, run_name: str) -> Path:
    if not run_name:
        run_name = time.strftime("%Y%m%d_%H%M%S")
    run_dir = unet_dir / "runs" / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "examples").mkdir(parents=True, exist_ok=True)
    return run_dir


def gpu_check(run_dir: Path):
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
        lines.append("GPU не обнаружена: обучение будет на CPU (очень медленно).")

    text = "\n".join(lines)
    print("\n=== GPU CHECK ===")
    print(text)
    print("=================\n")
    (run_dir / "gpu_info.txt").write_text(text, encoding="utf-8")


def save_history_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_history(history_csv: Path, out_dir: Path):
    df = pd.read_csv(history_csv)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["train_dice"], label="train_dice")
    plt.plot(df["epoch"], df["val_dice"], label="val_dice")
    plt.xlabel("epoch"); plt.ylabel("dice"); plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "dice.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["train_f1"], label="train_f1")
    plt.plot(df["epoch"], df["val_f1"], label="val_f1")
    plt.xlabel("epoch"); plt.ylabel("f1"); plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "f1.png", dpi=150)
    plt.close()


def resolve_path(p: str) -> str:
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((project_root() / pp).resolve())


def load_manifest(cfg: Config) -> pd.DataFrame:
    p = project_root() / cfg.manifest_rel
    if not p.exists():
        raise FileNotFoundError(f"Не найден manifest: {p}")

    df = pd.read_csv(p)
    need = ["case_id", "image_path", "mask_path"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Manifest должен содержать {need}. Сейчас: {df.columns.tolist()}")

    df["case_id"] = df["case_id"].astype(str).str.strip()
    df["image_path"] = df["image_path"].astype(str).map(resolve_path)
    df["mask_path"] = df["mask_path"].astype(str).map(resolve_path)

    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.lower().str.strip()

    return df


def make_splits(cfg: Config, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "split" not in df.columns:
        ids = df["case_id"].tolist()
        tr_ids, te_ids = train_test_split(ids, test_size=cfg.test_size, random_state=cfg.seed, shuffle=True)
        tr_set = set(tr_ids)
        df = df.copy()
        df["split"] = df["case_id"].apply(lambda x: "train" if x in tr_set else "test")

    if "val" not in set(df["split"].unique()):
        tr_df = df[df["split"] == "train"].copy()
        val_df = tr_df.sample(frac=cfg.val_from_train, random_state=cfg.seed)
        df.loc[val_df.index, "split"] = "val"

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df


def make_items(df: pd.DataFrame) -> List[Dict]:
    return [{"image": r["image_path"], "label": r["mask_path"], "case_id": r["case_id"]} for _, r in df.iterrows()]


def safe_nan_to_num_np(x):
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


# -------------------------- TRANSFORMS (FIXED) -------------------------- #
def build_transforms(cfg: Config):
    """
    Ключевой фикс:
    - Spacingd применяем ТОЛЬКО к image
    - label ресемплим строго под image через ResampleToMatchd
    Это устраняет рассинхрон размеров, из-за которого падал RandCropByPosNegLabeld.
    """
    base = [
        transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
    ]

    if cfg.use_orientation_ras:
        base += [transforms.Orientationd(keys=["image", "label"], axcodes="RAS")]

    base += [
        transforms.Spacingd(
            keys=["image"],
            pixdim=cfg.pixdim,
            mode=("bilinear",),
        ),
        transforms.ResampleToMatchd(
            keys=["label"],
            key_dst="image",
            mode="nearest",
        ),
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


# -------------------------- METRICS -------------------------- #
def binary_f1_from_logits(logits_2ch: torch.Tensor, labels_1ch: torch.Tensor, eps=1e-8) -> float:
    pred = torch.argmax(logits_2ch, dim=1).float()          # [B,D,H,W]
    gt = (labels_1ch[:, 0] > 0).float()                     # [B,D,H,W]

    tp = torch.sum(pred * gt)
    fp = torch.sum(pred * (1 - gt))
    fn = torch.sum((1 - pred) * gt)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return float(f1.item())


def three_view_overlay_png(out_png: Path, img3d: np.ndarray, gt3d: np.ndarray, pr3d: np.ndarray, title: str):
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
    if cfg.use_basic_unet:
        model = BasicUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            features=(32, 64, 128, 256, 512, 32),
            dropout=0.1,
        ).to(device)
        print("Используется BasicUNet (как в ноутбуке).")
        return model

    if cfg.roi[0] == 64:
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64),
            strides=(2, 2),
            num_res_units=2,
            kernel_size=3,
            up_kernel_size=3,
            norm="batch",
        ).to(device)
        print("Используется UNet для ROI=64 (как в ноутбуке).")
        return model

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        kernel_size=3,
        up_kernel_size=3,
        norm="batch",
    ).to(device)
    print("Используется UNet (как в ноутбуке).")
    return model


# -------------------------- TRAIN / EVAL -------------------------- #
def train_one_epoch(model, loader, optimizer, dice_loss, ce_loss, cfg, device, scaler, post_pred, post_label, dice_metric):
    model.train()
    epoch_loss = 0.0
    n_steps = 0

    train_dice_vals = []
    train_f1_vals = []

    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["image"].to(device)   # [B,1,D,H,W]
        y = batch["label"].to(device)   # [B,1,D,H,W]

        y_bin = (y > 0).long()

        optimizer.zero_grad(set_to_none=True)

        use_amp = cfg.amp and torch.cuda.is_available()
        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(x)  # [B,2,D,H,W]
            loss = cfg.dice_weight * dice_loss(logits, y_bin)
            if cfg.ce_weight > 0:
                loss = loss + cfg.ce_weight * ce_loss(logits, y_bin[:, 0])

        if not torch.isfinite(loss):
            print("[WARN] non-finite loss -> skip batch")
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

            f1 = binary_f1_from_logits(logits, y)
            train_f1_vals.append(f1)

    epoch_loss = epoch_loss / max(1, n_steps)
    train_dice = float(np.mean(train_dice_vals)) if train_dice_vals else 0.0
    train_f1 = float(np.mean(train_f1_vals)) if train_f1_vals else 0.0
    return epoch_loss, train_dice, train_f1


@torch.no_grad()
def evaluate(model, loader, dice_loss, ce_loss, cfg, device, post_pred, post_label, dice_metric):
    model.eval()
    val_losses = []
    val_f1_vals = []

    batches = 0
    for batch in tqdm(loader, desc="val", leave=False):
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        y_bin = (y > 0).long()

        logits = model(x)

        loss = cfg.dice_weight * dice_loss(logits, y_bin)
        if cfg.ce_weight > 0:
            loss = loss + cfg.ce_weight * ce_loss(logits, y_bin[:, 0])
        val_losses.append(float(loss.item()))

        preds = [post_pred(i) for i in decollate_batch(logits)]
        labs = [post_label(i) for i in decollate_batch(y_bin)]
        dice_metric(y_pred=preds, y=labs)

        f1 = binary_f1_from_logits(logits, y)
        val_f1_vals.append(f1)

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
    model.eval()
    _, val_tf = build_transforms(cfg)

    ex_dir = run_dir / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)

    pick = df_test.sample(n=min(cfg.examples_n, len(df_test)), random_state=cfg.seed).reset_index(drop=True)

    for i in range(len(pick)):
        case_id = str(pick.loc[i, "case_id"])
        img_path = str(pick.loc[i, "image_path"])
        msk_path = str(pick.loc[i, "mask_path"])

        sample = val_tf({"image": img_path, "label": msk_path})
        x = sample["image"].unsqueeze(0).to(device)   # [1,1,D,H,W]
        y = sample["label"].unsqueeze(0).to(device)   # [1,1,D,H,W]

        logits = model(x)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)  # [D,H,W]
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
    cfg = Config()
    set_seed(cfg.seed)

    unet_dir = Path(__file__).resolve().parent
    run_dir = ensure_run_dir(unet_dir, cfg.run_name)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_check(run_dir)

    df = load_manifest(cfg)
    train_df, val_df, test_df = make_splits(cfg, df)

    print(f"Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")
    pd.concat([
        train_df.assign(split="train"),
        val_df.assign(split="val"),
        test_df.assign(split="test"),
    ]).to_csv(run_dir / "manifest_used.csv", index=False)

    train_items = make_items(train_df)
    val_items = make_items(val_df)
    test_items = make_items(test_df)

    train_tf, val_tf = build_transforms(cfg)

    # train лучше НЕ кэшировать полностью из-за рандом-кропа, но cache_rate=0.2 допустимо (как у тебя)
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
    ce_loss = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, cfg.pos_class_weight], device=device)
    )

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = GradScaler("cuda", enabled=(cfg.amp and torch.cuda.is_available()))

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    history = []
    history_csv = run_dir / "history.csv"
    best_dice = -1.0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        tr_loss, tr_dice, tr_f1 = train_one_epoch(
            model, train_loader, optimizer, dice_loss, ce_loss, cfg, device, scaler,
            post_pred, post_label, dice_metric
        )

        val_loss, val_dice, val_f1 = evaluate(
            model, val_loader, dice_loss, ce_loss, cfg, device,
            post_pred, post_label, dice_metric
        )

        dt = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "train_dice": tr_dice,
            "val_dice": val_dice,
            "train_f1": tr_f1,
            "val_f1": val_f1,
            "lr": optimizer.param_groups[0]["lr"],
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
            f"time={dt:.1f}s"
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
        model, test_loader, dice_loss, ce_loss, cfg, device,
        post_pred, post_label, dice_metric
    )
    print(f"TEST loss={test_loss:.4f} dice={test_dice:.4f} f1={test_f1:.4f}")
    pd.DataFrame([{"test_loss": test_loss, "test_dice": test_dice, "test_f1": test_f1}]).to_csv(
        run_dir / "test_metrics.csv", index=False
    )

    print("\n=== Examples ===")
    save_examples(model, test_df, run_dir, cfg, device)

    print("\nDONE. Run dir:", run_dir.resolve())
    print("Artifacts:")
    print(" - history.csv + plots/*.png")
    print(" - checkpoints/best.pt, checkpoints/last.pt, checkpoints/best_metric_model.pth")
    print(" - examples/*.png + examples/*.nii.gz")


if __name__ == "__main__":
    main()
