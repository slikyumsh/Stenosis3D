"""
GT vs SegResNet predictions on data/test (images+masks folders), no manifest.

This script:
1) Loads SegResNet checkpoint (best.pt by default).
2) Iterates over data/test/images and finds matching masks in data/test/masks.
3) Applies the same val/test preprocessing style as your training script:
   LoadImaged -> (optional) Orientation RAS -> Spacing(image) -> ResampleToMatch(label)
   -> SpatialPad -> NormalizeIntensity -> nan_to_num -> EnsureTyped
4) Runs sliding-window inference (same pattern as training forward_for_eval).
5) Saves predicted masks (.nii.gz).
6) Builds meshes (PLY) for GT and predictions (marching cubes).
7) Builds centerlines using 3D skeletonization (skimage.morphology.skeletonize(method="lee")).
8) Estimates diameters along centerlines using EDT with sampling=pixdim.
9) Compares GT vs prediction diameter distributions via quantile profiles and computes MSE.
10) Saves per-case metrics and summary.

Install:
    pip install numpy pandas nibabel scipy scikit-image monai torch

Run:
    python experiments/segresnet_coronary_mse.py --out-dir experiments/exp01 --keep-largest-cc

Defaults:
    data dir: data/test (expects subfolders images/ and masks/)
    ckpt    : models/SegResNet/runs/20260114_183930/checkpoints/best.pt

Notes:
- Evaluation/export happens in PREPROCESSED SPACE (resampled/padded),
  to match the training-time preprocessing logic.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn as nn
from torch.amp import autocast

from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage import measure

from monai import transforms
from monai.transforms import Compose
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet


# ----------------------------- Config ----------------------------- #
@dataclass
class RunConfig:
    """Minimal config needed for inference + measurement."""
    roi: Tuple[int, int, int] = (96, 96, 96)
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    use_orientation_ras: bool = True
    sw_batch_size: int = 1
    sw_overlap: float = 0.25
    amp: bool = True


def load_run_config(run_dir: Path) -> RunConfig:
    """Load config.json from run_dir if present, otherwise use defaults."""
    cfg = RunConfig()
    p = run_dir / "config.json"
    if not p.exists():
        return cfg
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return cfg

    if "roi" in data:
        cfg.roi = tuple(data["roi"])
    if "pixdim" in data:
        cfg.pixdim = tuple(data["pixdim"])
    if "use_orientation_ras" in data:
        cfg.use_orientation_ras = bool(data["use_orientation_ras"])
    if "sw_batch_size" in data:
        cfg.sw_batch_size = int(data["sw_batch_size"])
    if "sw_overlap" in data:
        cfg.sw_overlap = float(data["sw_overlap"])
    if "amp" in data:
        cfg.amp = bool(data["amp"])
    return cfg


# ----------------------------- Helpers ----------------------------- #
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_nan_to_num_np(x):
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def strip_nii_ext(name: str) -> str:
    name = str(name)
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    if name.lower().endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def normalize_key(k: str) -> str:
    """
    Make matching between image and mask filenames more robust:
    - lower
    - remove common prefixes: mask_, seg_, label_
    - remove common suffixes: _mask, _masks, _seg, _segmentation, _label, _labels, _gt, _truth
    """
    k = strip_nii_ext(k).lower()

    for pref in ("mask_", "seg_", "label_", "gt_"):
        if k.startswith(pref):
            k = k[len(pref):]

    suffixes = (
        "_mask", "_masks",
        "_seg", "_segs", "_segmentation",
        "_label", "_labels",
        "_gt", "_truth",
    )
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if k.endswith(suf):
                k = k[: -len(suf)]
                changed = True
                break

    return k.strip()


def list_nii_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    files = []
    files += sorted(folder.glob("*.nii"))
    files += sorted(folder.glob("*.nii.gz"))
    return files


def pair_images_and_masks(images_dir: Path, masks_dir: Path) -> List[Tuple[str, Path, Path]]:
    """
    Return list of (case_id, image_path, mask_path).
    Matching strategy:
      1) exact base name match
      2) normalized base name match (strip prefixes/suffixes)
    """
    images = list_nii_files(images_dir)
    masks = list_nii_files(masks_dir)

    mask_exact: Dict[str, Path] = {}
    mask_norm: Dict[str, List[Path]] = {}

    for mp in masks:
        base = strip_nii_ext(mp.name)
        base_norm = normalize_key(base)
        mask_exact[base] = mp
        mask_exact[base_norm] = mp  # allow direct lookup by normalized key too
        mask_norm.setdefault(base_norm, []).append(mp)

    pairs: List[Tuple[str, Path, Path]] = []
    missing_masks: List[str] = []

    for ip in images:
        base = strip_nii_ext(ip.name)
        key_norm = normalize_key(base)

        mp = None
        if base in mask_exact:
            mp = mask_exact[base]
        elif key_norm in mask_exact:
            mp = mask_exact[key_norm]
        elif key_norm in mask_norm and len(mask_norm[key_norm]) > 0:
            mp = sorted(mask_norm[key_norm])[0]

        if mp is None:
            missing_masks.append(ip.name)
            continue

        case_id = key_norm if key_norm else base
        pairs.append((case_id, ip, mp))

    if missing_masks:
        print(f"[WARN] no matching mask for {len(missing_masks)} images (showing up to 10):")
        for n in missing_masks[:10]:
            print("   -", n)

    return pairs


# ----------------------------- MONAI transforms ----------------------------- #
def build_val_transforms(cfg: RunConfig):
    """
    Mirrors your training val/test preprocessing.
    """
    base = [
        transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
    ]
    if cfg.use_orientation_ras:
        base += [transforms.Orientationd(keys=["image", "label"], axcodes="RAS", labels=None)]

    base += [
        transforms.Spacingd(keys=["image"], pixdim=cfg.pixdim, mode=("bilinear",)),
        transforms.ResampleToMatchd(keys=["label"], key_dst="image", mode="nearest"),
        transforms.SpatialPadd(keys=["image", "label"], spatial_size=cfg.roi),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        transforms.Lambdad(keys=["image"], func=safe_nan_to_num_np),
        transforms.EnsureTyped(keys=["image", "label"]),
    ]
    return Compose(base)


# ----------------------------- Model / inference ----------------------------- #
def build_segresnet(device: torch.device) -> nn.Module:
    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=32,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.1,
    ).to(device)
    return model


def load_checkpoint_payload(ckpt_path: Path) -> Dict:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(str(ckpt_path), map_location="cpu")


def load_model_weights(model: nn.Module, ckpt_payload: Dict) -> None:
    if isinstance(ckpt_payload, dict) and "model_state_dict" in ckpt_payload:
        model.load_state_dict(ckpt_payload["model_state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt_payload, strict=True)


@torch.no_grad()
def forward_for_eval(cfg: RunConfig, model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    # exactly the same idea as in training script
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


@torch.no_grad()
def predict_mask(model: nn.Module, x: torch.Tensor, cfg: RunConfig, device: torch.device) -> np.ndarray:
    use_amp = cfg.amp and device.type == "cuda"
    device_type = "cuda" if device.type == "cuda" else "cpu"
    with autocast(device_type=device_type, enabled=use_amp):
        logits = forward_for_eval(cfg, model, x, device)
    pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    return pred


# ----------------------------- Measurements ----------------------------- #
def binarize(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def keep_largest_cc_6n(mask_bin: np.ndarray) -> np.ndarray:
    from collections import deque

    mask_bin = (mask_bin > 0).astype(np.uint8)
    visited = np.zeros(mask_bin.shape, dtype=np.uint8)

    neigh = [(1, 0, 0), (-1, 0, 0),
             (0, 1, 0), (0, -1, 0),
             (0, 0, 1), (0, 0, -1)]

    best_voxels = None
    best_count = 0

    starts = np.argwhere(mask_bin > 0)
    for sx, sy, sz in starts:
        if visited[sx, sy, sz]:
            continue

        q = deque([(sx, sy, sz)])
        visited[sx, sy, sz] = 1
        voxels = [(sx, sy, sz)]

        while q:
            x, y, z = q.popleft()
            for dx, dy, dz in neigh:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < mask_bin.shape[0] and 0 <= ny < mask_bin.shape[1] and 0 <= nz < mask_bin.shape[2]:
                    if mask_bin[nx, ny, nz] and not visited[nx, ny, nz]:
                        visited[nx, ny, nz] = 1
                        q.append((nx, ny, nz))
                        voxels.append((nx, ny, nz))

        if len(voxels) > best_count:
            best_count = len(voxels)
            best_voxels = voxels

    out = np.zeros_like(mask_bin, dtype=np.uint8)
    if best_voxels is not None:
        xs, ys, zs = zip(*best_voxels)
        out[np.array(xs), np.array(ys), np.array(zs)] = 1
    return out


def skeleton_centerline(mask_bin: np.ndarray) -> np.ndarray:
    if mask_bin.max() == 0:
        return np.zeros_like(mask_bin, dtype=np.uint8)
    sk = skeletonize(mask_bin.astype(bool), method="lee")
    return sk.astype(np.uint8)


def edt_radius_mm(mask_bin: np.ndarray, pixdim: Tuple[float, float, float]) -> np.ndarray:
    return distance_transform_edt(mask_bin.astype(bool), sampling=pixdim).astype(np.float32)


def centerline_diameters(mask_bin: np.ndarray, pixdim: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    sk = skeleton_centerline(mask_bin)
    pts = np.argwhere(sk > 0)
    if pts.size == 0:
        return pts.astype(np.int32), np.array([], dtype=np.float32)

    r = edt_radius_mm(mask_bin, pixdim)
    diam = (2.0 * r[pts[:, 0], pts[:, 1], pts[:, 2]]).astype(np.float32)

    good = np.isfinite(diam) & (diam > 0)
    return pts[good].astype(np.int32), diam[good].astype(np.float32)


def mask_to_mesh_ply(mask_bin: np.ndarray, pixdim: Tuple[float, float, float], out_ply: Path) -> None:
    if mask_bin.max() == 0:
        out_ply.write_text(
            "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\n"
            "element face 0\nproperty list uchar int vertex_indices\nend_header\n",
            encoding="utf-8",
        )
        return

    verts, faces, _, _ = measure.marching_cubes(mask_bin.astype(np.uint8), level=0.5, spacing=pixdim)

    with open(out_ply, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def quantile_profile(x: np.ndarray, n: int) -> np.ndarray:
    if x.size == 0:
        return np.zeros((n,), dtype=np.float32)
    qs = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.quantile(x.astype(np.float32), qs).astype(np.float32)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean((a - b) ** 2))


def save_points_csv(path: Path, pts: np.ndarray, diam: np.ndarray) -> None:
    df = pd.DataFrame({
        "x": pts[:, 0] if pts.size else [],
        "y": pts[:, 1] if pts.size else [],
        "z": pts[:, 2] if pts.size else [],
        "diameter_mm": diam if diam.size else [],
    })
    df.to_csv(path, index=False)


def save_uint8_nifti(data_u8: np.ndarray, out_path: Path, pixdim: Tuple[float, float, float]) -> None:
    affine = np.eye(4, dtype=np.float32)
    affine[0, 0] = float(pixdim[0])
    affine[1, 1] = float(pixdim[1])
    affine[2, 2] = float(pixdim[2])
    nii = nib.Nifti1Image(data_u8.astype(np.uint8), affine=affine)
    nii.set_data_dtype(np.uint8)
    nib.save(nii, str(out_path))


# ----------------------------- Output layout ----------------------------- #
@dataclass
class OutPaths:
    root: Path
    pred_masks: Path
    gt_mesh: Path
    pr_mesh: Path
    gt_centerline_nii: Path
    pr_centerline_nii: Path
    gt_centerline_csv: Path
    pr_centerline_csv: Path
    metrics: Path


def make_out_dirs(out_dir: Path) -> OutPaths:
    p = OutPaths(
        root=out_dir,
        pred_masks=out_dir / "pred_masks",
        gt_mesh=out_dir / "gt_mesh",
        pr_mesh=out_dir / "pred_mesh",
        gt_centerline_nii=out_dir / "gt_centerline_nii",
        pr_centerline_nii=out_dir / "pred_centerline_nii",
        gt_centerline_csv=out_dir / "gt_centerline_csv",
        pr_centerline_csv=out_dir / "pred_centerline_csv",
        metrics=out_dir / "metrics",
    )
    for d in [
        p.pred_masks, p.gt_mesh, p.pr_mesh,
        p.gt_centerline_nii, p.pr_centerline_nii,
        p.gt_centerline_csv, p.pr_centerline_csv,
        p.metrics,
    ]:
        ensure_dir(d)
    return p


# ----------------------------- Main ----------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True, help="Experiment output directory.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path("data") / "test"),
        help="Test data directory (expects subfolders images/ and masks/). Default: data/test",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(Path("models") / "SegResNet" / "runs" / "20260114_183930" / "checkpoints" / "best.pt"),
        help="Checkpoint path (best.pt).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of cases (0 = all).")
    parser.add_argument("--n-quantiles", type=int, default=1000, help="Quantiles for diameter profile MSE.")
    parser.add_argument("--keep-largest-cc", action="store_true", help="Keep only the largest CC before measurements.")
    parser.add_argument("--device", type=str, default="", help="Device override: 'cuda' or 'cpu' (default: auto).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out = make_out_dirs(out_dir)

    data_dir = Path(args.data_dir)
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"

    if not images_dir.exists():
        raise FileNotFoundError(f"images dir not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"masks dir not found: {masks_dir}")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Infer run_dir for config.json
    run_dir = ckpt_path.parent.parent  # .../runs/<run>/checkpoints/best.pt -> .../runs/<run>
    cfg = load_run_config(run_dir)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    ckpt_payload = load_checkpoint_payload(ckpt_path)
    model = build_segresnet(device)
    load_model_weights(model, ckpt_payload)
    model.eval()

    # Transforms
    val_tf = build_val_transforms(cfg)

    # Collect pairs
    pairs = pair_images_and_masks(images_dir, masks_dir)
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]

    print("DATA DIR   :", data_dir.resolve())
    print("IMAGES DIR :", images_dir.resolve())
    print("MASKS DIR  :", masks_dir.resolve())
    print("CKPT       :", ckpt_path.resolve())
    print("RUN DIR    :", run_dir.resolve())
    print("DEVICE     :", device)
    print("PIXDIM     :", cfg.pixdim)
    print("ROI        :", cfg.roi)
    print("CASES      :", len(pairs))
    print("OUT DIR    :", out.root.resolve())
    print()

    # Save config snapshot
    (out.root / "experiment_config.json").write_text(
        json.dumps(
            {
                "data_dir": str(data_dir),
                "images_dir": str(images_dir),
                "masks_dir": str(masks_dir),
                "checkpoint": str(ckpt_path),
                "run_dir_inferred": str(run_dir),
                "pixdim": cfg.pixdim,
                "roi": cfg.roi,
                "sw_batch_size": cfg.sw_batch_size,
                "sw_overlap": cfg.sw_overlap,
                "amp": cfg.amp,
                "keep_largest_cc": bool(args.keep_largest_cc),
                "n_quantiles": int(args.n_quantiles),
                "device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows: List[Dict] = []
    for i, (case_id, img_path, gt_mask_path) in enumerate(pairs):
        print(f"[{i+1:04d}/{len(pairs):04d}] case_id={case_id}")

        try:
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            if not gt_mask_path.exists():
                raise FileNotFoundError(f"Mask not found: {gt_mask_path}")

            # --- 1) Load + preprocess (like training val/test) ---
            sample = val_tf({"image": str(img_path), "label": str(gt_mask_path)})

            x = sample["image"].unsqueeze(0).to(device)  # [1,1,D,H,W]
            gt_t = sample["label"]
            if gt_t.ndim == 5:
                gt = gt_t[0, 0].cpu().numpy()
            elif gt_t.ndim == 4:
                gt = gt_t[0].cpu().numpy()
            else:
                gt = gt_t.cpu().numpy()

            gt_bin = binarize(gt)

            # --- 2) Predict (sliding-window) ---
            pred = predict_mask(model, x, cfg, device)
            pr_bin = binarize(pred)

            if args.keep_largest_cc:
                gt_bin = keep_largest_cc_6n(gt_bin)
                pr_bin = keep_largest_cc_6n(pr_bin)

            # --- 3) Save predicted mask NIfTI ---
            pred_nii_path = out.pred_masks / f"case_{case_id}_pred.nii.gz"
            save_uint8_nifti(pr_bin, pred_nii_path, cfg.pixdim)

            # --- 4) Mesh export ---
            gt_ply = out.gt_mesh / f"case_{case_id}_gt.ply"
            pr_ply = out.pr_mesh / f"case_{case_id}_pred.ply"
            mask_to_mesh_ply(gt_bin, cfg.pixdim, gt_ply)
            mask_to_mesh_ply(pr_bin, cfg.pixdim, pr_ply)

            # --- 5) Centerlines ---
            gt_skel = skeleton_centerline(gt_bin)
            pr_skel = skeleton_centerline(pr_bin)

            gt_skel_path = out.gt_centerline_nii / f"case_{case_id}_gt_centerline.nii.gz"
            pr_skel_path = out.pr_centerline_nii / f"case_{case_id}_pred_centerline.nii.gz"
            save_uint8_nifti(gt_skel, gt_skel_path, cfg.pixdim)
            save_uint8_nifti(pr_skel, pr_skel_path, cfg.pixdim)

            # --- 6) Diameters ---
            gt_pts, gt_d = centerline_diameters(gt_bin, cfg.pixdim)
            pr_pts, pr_d = centerline_diameters(pr_bin, cfg.pixdim)

            gt_csv = out.gt_centerline_csv / f"case_{case_id}_gt_centerline.csv"
            pr_csv = out.pr_centerline_csv / f"case_{case_id}_pred_centerline.csv"
            save_points_csv(gt_csv, gt_pts, gt_d)
            save_points_csv(pr_csv, pr_pts, pr_d)

            # --- 7) Compare ---
            gt_prof = quantile_profile(gt_d, n=int(args.n_quantiles))
            pr_prof = quantile_profile(pr_d, n=int(args.n_quantiles))
            m = mse(gt_prof, pr_prof)

            def stats(x_: np.ndarray) -> Tuple[float, float, float]:
                if x_.size == 0:
                    return 0.0, 0.0, 0.0
                return float(np.mean(x_)), float(np.median(x_)), float(np.max(x_))

            gt_mean, gt_med, gt_max = stats(gt_d)
            pr_mean, pr_med, pr_max = stats(pr_d)

            rows.append({
                "case_id": case_id,
                "image_path": str(img_path),
                "gt_mask_path": str(gt_mask_path),
                "pred_mask_path": str(pred_nii_path),
                "pixdim_x": float(cfg.pixdim[0]),
                "pixdim_y": float(cfg.pixdim[1]),
                "pixdim_z": float(cfg.pixdim[2]),
                "gt_centerline_points": int(gt_d.size),
                "pred_centerline_points": int(pr_d.size),
                "gt_diam_mean_mm": gt_mean,
                "gt_diam_median_mm": gt_med,
                "gt_diam_max_mm": gt_max,
                "pred_diam_mean_mm": pr_mean,
                "pred_diam_median_mm": pr_med,
                "pred_diam_max_mm": pr_max,
                "mse_diameter_quantiles": float(m),
                "gt_mesh_ply": str(gt_ply),
                "pred_mesh_ply": str(pr_ply),
                "gt_centerline_nii": str(gt_skel_path),
                "pred_centerline_nii": str(pr_skel_path),
                "gt_centerline_csv": str(gt_csv),
                "pred_centerline_csv": str(pr_csv),
            })

            print(f"  saved pred: {pred_nii_path.name}")
            print(f"  mse_diameter_quantiles = {m:.6f}")

        except Exception as e:
            print(f"  [ERROR] case_id={case_id}")
            print(f"         image: {img_path}")
            print(f"         label: {gt_mask_path}")
            print(f"         exc  : {repr(e)}")

            # Extra: nibabel load to reveal file-format errors clearly
            try:
                _ = nib.load(str(img_path))
            except Exception as e_img:
                print(f"         nib.load(image) failed: {repr(e_img)}")
            try:
                _ = nib.load(str(gt_mask_path))
            except Exception as e_lbl:
                print(f"         nib.load(label) failed: {repr(e_lbl)}")

            traceback.print_exc()
            rows.append({
                "case_id": case_id,
                "image_path": str(img_path),
                "gt_mask_path": str(gt_mask_path),
                "error": repr(e),
            })

    df_out = pd.DataFrame(rows)
    per_case_csv = out.metrics / "per_case_metrics.csv"
    df_out.to_csv(per_case_csv, index=False)

    ok = df_out[df_out.get("error").isna()] if "error" in df_out.columns else df_out
    summary = {
        "num_cases_total": int(len(df_out)),
        "num_cases_ok": int(len(ok)),
        "num_cases_failed": int(len(df_out) - len(ok)),
        "mse_mean": float(ok["mse_diameter_quantiles"].mean()) if len(ok) and "mse_diameter_quantiles" in ok.columns else 0.0,
        "mse_median": float(ok["mse_diameter_quantiles"].median()) if len(ok) and "mse_diameter_quantiles" in ok.columns else 0.0,
    }
    summary_csv = out.metrics / "summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    print("\nDONE")
    print("Per-case metrics:", per_case_csv.resolve())
    print("Summary         :", summary_csv.resolve())
    print("Artifacts:")
    print(" - pred masks       :", out.pred_masks.resolve())
    print(" - gt mesh          :", out.gt_mesh.resolve())
    print(" - pred mesh        :", out.pr_mesh.resolve())
    print(" - gt centerline nii:", out.gt_centerline_nii.resolve())
    print(" - pred centerline  :", out.pr_centerline_nii.resolve())
    print(" - metrics          :", out.metrics.resolve())


if __name__ == "__main__":
    main()
