# Stenosis3D — 3D Coronary Vessel Segmentation + Geometry-Based Stenosis Analysis + 2D Projection Detection

This repository contains code for a research pipeline that:
1) segments coronary vessels in 3D volumes using a 3D SegResNet,
2) extracts vessel centerlines and estimates local diameters from the segmentation,
3) optionally localizes stenosis candidates using **2D detection** on orthogonal projections.

The core idea is to combine **deep 3D segmentation** (robust vessel extraction) with **interpretable geometry** (centerline + diameter profile) and a **lightweight stenosis-localization module** (YOLO on projections), enabling both quantitative evaluation and “paper-ready” visualizations.

---

## Repository layout

Top-level structure (main branch):

- `data/`  
  Dataset assets and/or local dataset organization. Recommended split:
  - `data/train/images`, `data/train/masks`
  - `data/val/images`, `data/val/masks`
  - `data/test/images`, `data/test/masks`

- `models/`  
  Training runs, configs, and checkpoints (e.g., SegResNet runs).

- `scripts/`  
  Scripts for training, evaluation, experiments, and visualization utilities.

- `tools/`  
  Helper utilities (I/O, transforms, geometry, visualization).

- `__main__.py`  
  The main orchestrator script used for end-to-end experiments / inference / evaluation.

- `requirements.txt`  
  Python dependencies.

---

## Method overview

### 1) 3D vessel segmentation (SegResNet)

We use a 3D SegResNet-style architecture implemented in MONAI.

Typical steps:
- Load NIfTI image/mask
- Optional reorientation to RAS
- Spacing normalization (resampling to a target voxel spacing)
- Patch-based training with random ROI crops
- Sliding-window inference for full-volume evaluation

This avoids common size-mismatch issues for full-volume inference and keeps results consistent across scanners/resolutions.

### 2) Centerline extraction

Given a binary vessel mask, we compute a centerline proxy using 3D skeletonization (thin medial representation).  
This preserves vessel topology and gives a sparse representation suitable for geometric measurements.

### 3) Diameter estimation from Euclidean Distance Transform (EDT)

We estimate vessel radius in millimeters using a distance transform computed inside the binary vessel mask.

For each centerline voxel:
- `radius_mm = EDT(mask, sampling=pixdim)[centerline_voxel]`
- `diameter_mm = 2 * radius_mm`

This produces an interpretable diameter profile along the vessel tree.

### 4) Stenosis candidate localization

We support two complementary routes:

**A) 3D geometry-based candidates (diameter profile analysis)**  
Stenosis can be treated as a localized drop in diameter along the centerline. Candidate regions can be detected by:
- relative drop vs. a local baseline,
- analyzing local minima / slopes,
- or distribution-based comparisons between GT and predictions (quantile profiles).

**B) 2D detection on orthogonal projections (YOLOv8)**  
We compute three orthogonal 2D projections from a 3D volume (e.g., MIP/sum projection) and run YOLOv8 to predict bounding boxes of potential stenosis in each view.  
These boxes can be fused back to a 3D ROI (view-consistent back-projection) if needed.

---

## Visualizations

The pipeline supports “paper-ready” visuals, including:
- Vessel surface mesh (marching cubes)
- Centerline overlays (colored polylines/tubes)
- Example markers (spheres) placed along the centerline
- Three orthogonal projection panels with YOLO bounding boxes

---

## Installation

Create and activate a virtual environment, then:

```bash
pip install -r requirements.txt
```

---

## Running

### Segmentation inference / evaluation

1) Place your data in:

```text
data/test/images
data/test/masks
```

2) Use a trained checkpoint, e.g.:

```text
models/SegResNet/runs/<RUN_ID>/checkpoints/best.pt
```

3) Run the pipeline (see `--help` for available args):

```bash
python __main__.py --help
```

> Scripts auto-select GPU if available; otherwise they fall back to CPU.

### 2D projection detection (YOLO)

If you have a YOLOv8 detector in `detector/`:
- `detector/best.pt`
- `detector/data.yaml` (optional)

You can run detection on orthogonal projections and save a montage of the process.

---
