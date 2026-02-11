from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

from skimage.morphology import skeletonize

try:
    import yaml
except Exception:
    yaml = None

from ultralytics import YOLO


# ----------------------------- IO ----------------------------- #
def load_nii_mask(path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """
    Returns:
      mask_u8  : [X,Y,Z] uint8 0/1
      affine   : 4x4
      pixdim   : (dx,dy,dz) mm
    """
    nii = nib.load(str(path))
    data = nii.get_fdata(dtype=np.float32)
    mask = (data > 0).astype(np.uint8)
    affine = nii.affine
    zooms = nii.header.get_zooms()
    pixdim = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return mask, affine, pixdim


def save_nii_u8(path: Path, data_u8: np.ndarray, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = nib.Nifti1Image(data_u8.astype(np.uint8), affine=affine)
    out.set_data_dtype(np.uint8)
    nib.save(out, str(path))


# ----------------------------- Projections ----------------------------- #
def mip_projection(mask_u8: np.ndarray, axis: int) -> np.ndarray:
    proj = np.max(mask_u8, axis=axis).astype(np.uint8)  # 0/1
    return (proj * 255).astype(np.uint8)


def make_three_projections(mask_u8: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "axial": mip_projection(mask_u8, axis=2),     # XY
        "coronal": mip_projection(mask_u8, axis=1),   # XZ
        "sagittal": mip_projection(mask_u8, axis=0),  # YZ
    }


def to_3ch(img_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)


# ----------------------------- Skeleton graph + longest path ----------------------------- #
def skeleton_points_and_edges_26(skel_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(skel_u8 > 0)
    if pts.size == 0:
        return pts.astype(np.int32), np.zeros((0, 2), dtype=np.int32)

    pts_list = [tuple(p) for p in pts]
    idx = {p: i for i, p in enumerate(pts_list)}

    # half offsets to avoid duplicate edges
    half_offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                if (dx > 0) or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz > 0):
                    half_offsets.append((dx, dy, dz))

    sx, sy, sz = skel_u8.shape
    edges = []
    for i, (x, y, z) in enumerate(pts_list):
        for dx, dy, dz in half_offsets:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < sx and 0 <= ny < sy and 0 <= nz < sz:
                j = idx.get((nx, ny, nz), -1)
                if j >= 0:
                    edges.append((i, j))

    if not edges:
        return pts.astype(np.int32), np.zeros((0, 2), dtype=np.int32)

    return pts.astype(np.int32), np.array(edges, dtype=np.int32)


def longest_path_on_skeleton_graph(pts_vox: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if pts_vox.size == 0:
        return pts_vox.astype(np.int32)
    n = pts_vox.shape[0]
    if edges.size == 0:
        return pts_vox[:1].astype(np.int32)

    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)

    degrees = np.array([len(a) for a in adj], dtype=np.int32)
    endpoints = np.where(degrees == 1)[0]
    start = int(endpoints[0]) if endpoints.size > 0 else 0

    def bfs(source: int):
        from collections import deque
        dist = np.full((n,), -1, dtype=np.int32)
        parent = np.full((n,), -1, dtype=np.int32)
        q = deque([source])
        dist[source] = 0
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
        far = int(np.argmax(dist))
        return far, dist, parent

    a, _, _ = bfs(start)
    b, _, parent = bfs(a)

    # reconstruct path b->a
    path_idx = []
    cur = b
    while cur != -1:
        path_idx.append(cur)
        if cur == a:
            break
        cur = int(parent[cur])
    path_idx = path_idx[::-1]
    return pts_vox[np.array(path_idx, dtype=np.int32)].astype(np.int32)


def vox_to_world(pts_vox: np.ndarray, pixdim: Tuple[float, float, float]) -> np.ndarray:
    return pts_vox.astype(np.float32) * np.array(pixdim, dtype=np.float32)[None, :]


# ----------------------------- Simulated stenosis (pinch) ----------------------------- #
def simulate_stenosis_pinch(
    mask_u8: np.ndarray,
    pixdim: Tuple[float, float, float],
    ratio: float = 0.35,
    half_length_mm: float = 7.0,
    center_mode: str = "middle",
) -> Tuple[np.ndarray, Dict]:
    """
    Create a synthetic stenosis by constraining vessel voxels inside a local cylinder
    around a chosen centerline point.

    - Build skeleton (Lee)
    - Extract main branch (longest path)
    - Pick center point:
        center_mode="middle" or "min_radius"
    - Estimate tangent direction from main path (world coords)
    - In a local slab (|t|<=half_length_mm), keep only voxels within
        perpendicular distance <= (radius_at_center * ratio)

    Returns:
      mask_new_u8, info_dict
    """
    mask = (mask_u8 > 0).astype(np.uint8)
    if mask.max() == 0:
        return mask, {"ok": False, "reason": "empty mask"}

    # skeleton
    skel = skeletonize(mask.astype(bool), method="lee").astype(np.uint8)
    pts, edges = skeleton_points_and_edges_26(skel)
    if pts.size == 0:
        return mask, {"ok": False, "reason": "empty skeleton"}

    main_path = longest_path_on_skeleton_graph(pts, edges)
    if main_path.size < 5:
        return mask, {"ok": False, "reason": "main path too short"}

    # EDT radius map (mm)
    r = distance_transform_edt(mask.astype(bool), sampling=pixdim).astype(np.float32)
    rr = r[main_path[:, 0], main_path[:, 1], main_path[:, 2]]
    rr = np.where(np.isfinite(rr), rr, 0.0)

    if center_mode == "min_radius":
        valid = rr > 0
        if np.any(valid):
            k0 = int(np.argmin(rr[valid]))
            k0 = int(np.where(valid)[0][k0])
        else:
            k0 = len(main_path) // 2
    else:
        k0 = len(main_path) // 2

    # tangent direction (world)
    main_w = vox_to_world(main_path, pixdim)

    w = 5  # window for tangent estimate
    k1 = max(0, k0 - w)
    k2 = min(len(main_w) - 1, k0 + w)
    v = main_w[k2] - main_w[k1]
    nv = float(np.linalg.norm(v))
    if nv < 1e-6:
        # fallback: use PCA-ish from endpoints
        v = main_w[-1] - main_w[0]
        nv = float(np.linalg.norm(v))
        if nv < 1e-6:
            return mask, {"ok": False, "reason": "cannot estimate direction"}
    v = v / nv  # unit direction

    center_vox = main_path[k0].astype(np.int32)
    center_w = main_w[k0].astype(np.float32)

    r0 = float(r[center_vox[0], center_vox[1], center_vox[2]])
    if r0 <= 0:
        r0 = float(np.max(rr)) if rr.size else 3.0
    target_radius = max(0.5, r0 * float(ratio))  # mm, clamp to not vanish

    # crop region around center to keep it fast
    margin_mm = max(half_length_mm, r0 * 2.0) + 3.0
    rad_vox = (
        int(np.ceil(margin_mm / pixdim[0])),
        int(np.ceil(margin_mm / pixdim[1])),
        int(np.ceil(margin_mm / pixdim[2])),
    )

    sx, sy, sz = mask.shape
    x0 = max(0, center_vox[0] - rad_vox[0]); x1 = min(sx, center_vox[0] + rad_vox[0] + 1)
    y0 = max(0, center_vox[1] - rad_vox[1]); y1 = min(sy, center_vox[1] + rad_vox[1] + 1)
    z0 = max(0, center_vox[2] - rad_vox[2]); z1 = min(sz, center_vox[2] + rad_vox[2] + 1)

    crop = mask[x0:x1, y0:y1, z0:z1].copy()
    if crop.max() == 0:
        return mask, {"ok": False, "reason": "crop empty"}

    # voxel grid -> world coords
    gx, gy, gz = np.indices(crop.shape, dtype=np.float32)
    gx += x0; gy += y0; gz += z0
    pts_w = np.stack([gx * pixdim[0], gy * pixdim[1], gz * pixdim[2]], axis=-1)  # [...,3]

    # vector from center
    d = pts_w - center_w[None, None, None, :]

    # along coordinate t and perpendicular distance
    t = d[..., 0] * v[0] + d[..., 1] * v[1] + d[..., 2] * v[2]  # dot
    perp = d - t[..., None] * v[None, None, None, :]
    d_perp = np.sqrt(np.sum(perp ** 2, axis=-1))

    # region where we apply stenosis
    in_slab = np.abs(t) <= float(half_length_mm)

    # keep only voxels close to centerline (cylinder)
    keep = d_perp <= float(target_radius)

    # apply only where vessel exists
    crop_new = crop.copy()
    apply = (crop > 0) & in_slab & (~keep)
    crop_new[apply] = 0

    mask_new = mask.copy()
    mask_new[x0:x1, y0:y1, z0:z1] = crop_new

    info = {
        "ok": True,
        "center_vox": [int(center_vox[0]), int(center_vox[1]), int(center_vox[2])],
        "center_mode": center_mode,
        "r0_mm": float(r0),
        "ratio": float(ratio),
        "target_radius_mm": float(target_radius),
        "half_length_mm": float(half_length_mm),
        "crop_bounds_xyz": [int(x0), int(x1), int(y0), int(y1), int(z0), int(z1)],
    }
    return mask_new.astype(np.uint8), info


# ----------------------------- YOLO helpers ----------------------------- #
def load_class_names(detector_dir: Path, model: YOLO) -> Dict[int, str]:
    try:
        names = model.names
        if isinstance(names, dict) and len(names) > 0:
            return {int(k): str(v) for k, v in names.items()}
    except Exception:
        pass

    yml = detector_dir / "data.yaml"
    if yaml is not None and yml.exists():
        try:
            d = yaml.safe_load(yml.read_text(encoding="utf-8"))
            if isinstance(d, dict) and "names" in d:
                nm = d["names"]
                if isinstance(nm, dict):
                    return {int(k): str(v) for k, v in nm.items()}
                if isinstance(nm, list):
                    return {i: str(v) for i, v in enumerate(nm)}
        except Exception:
            pass

    return {0: "stenosis"}


def run_yolo_on_image(model: YOLO, img_bgr: np.ndarray, conf: float, iou: float, device: str) -> List[Dict]:
    res = model.predict(img_bgr, conf=conf, iou=iou, device=device, verbose=False)
    r0 = res[0]
    dets: List[Dict] = []
    if r0.boxes is None or len(r0.boxes) == 0:
        return dets

    xyxy = r0.boxes.xyxy.detach().cpu().numpy()
    confs = r0.boxes.conf.detach().cpu().numpy()
    clss = r0.boxes.cls.detach().cpu().numpy()

    for b, c, k in zip(xyxy, confs, clss):
        dets.append({
            "xyxy": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
            "conf": float(c),
            "cls": int(k),
        })
    return dets


def draw_detections(img_bgr: np.ndarray, dets: List[Dict], class_names: Dict[int, str]) -> np.ndarray:
    out = img_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        conf = d["conf"]
        cls = d["cls"]
        label = f"{class_names.get(cls, str(cls))} {conf:.2f}"

        # red box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1, y_text), (x1 + tw + 4, y_text + th + 6), (0, 0, 255), -1)
        cv2.putText(out, label, (x1 + 2, y_text + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ----------------------------- Visualizations ----------------------------- #
def save_montage_before_after(out_png: Path, projs_before: Dict[str, np.ndarray], projs_after: Dict[str, np.ndarray]):
    keys = ["axial", "coronal", "sagittal"]
    plt.figure(figsize=(16, 7), dpi=160)

    for i, k in enumerate(keys):
        plt.subplot(2, 3, i + 1)
        plt.imshow(projs_before[k], cmap="gray")
        plt.title(f"{k} BEFORE")
        plt.axis("off")

        plt.subplot(2, 3, i + 4)
        plt.imshow(projs_after[k], cmap="gray")
        plt.title(f"{k} AFTER (sim stenosis)")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def save_montage_yolo(out_png: Path, projs_after: Dict[str, np.ndarray], det_imgs: Dict[str, np.ndarray]):
    keys = ["axial", "coronal", "sagittal"]
    plt.figure(figsize=(16, 7), dpi=160)

    for i, k in enumerate(keys):
        plt.subplot(2, 3, i + 1)
        plt.imshow(projs_after[k], cmap="gray")
        plt.title(f"{k} projection (AFTER)")
        plt.axis("off")

        plt.subplot(2, 3, i + 4)
        plt.imshow(cv2.cvtColor(det_imgs[k], cv2.COLOR_BGR2RGB))
        plt.title(f"{k} + YOLO boxes")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# ----------------------------- Main ----------------------------- #
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--mask", "-m", required=True, help="Path to 3D mask NIfTI (.nii/.nii.gz).")
    ap.add_argument("--detector-dir", "-d", default="detector", help="Folder with best.pt (and optional data.yaml).")
    ap.add_argument("--out-dir", "-o", required=True, help="Output directory.")

    # simulation params
    ap.add_argument("--simulate", action="store_true", help="Enable synthetic stenosis (pinch) on the mask.")
    ap.add_argument("--ratio", type=float, default=0.35, help="Stenosis radius ratio (smaller => stronger stenosis).")
    ap.add_argument("--half-length-mm", type=float, default=7.0, help="Half-length (mm) of stenosis region along vessel.")
    ap.add_argument("--center-mode", type=str, default="middle", choices=["middle", "min_radius"],
                    help="Where to place stenosis center on main branch.")

    # yolo params
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    ap.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold.")
    ap.add_argument("--device", type=str, default="", help="YOLO device: ''=auto, 'cpu', '0', '1', ...")

    args = ap.parse_args()

    mask_path = Path(args.mask)
    det_dir = Path(args.detector_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = det_dir / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Detector weights not found: {weights}")

    # load mask
    mask_u8, affine, pixdim = load_nii_mask(mask_path)

    # BEFORE projections
    projs_before = make_three_projections(mask_u8)
    for k, img in projs_before.items():
        cv2.imwrite(str(out_dir / f"proj_{k}_before.png"), img)

    # simulate stenosis if requested
    if args.simulate:
        mask_after, sim_info = simulate_stenosis_pinch(
            mask_u8=mask_u8,
            pixdim=pixdim,
            ratio=args.ratio,
            half_length_mm=args.half_length_mm,
            center_mode=args.center_mode,
        )
    else:
        mask_after = mask_u8.copy()
        sim_info = {"ok": False, "reason": "simulate disabled"}

    save_nii_u8(out_dir / "mask_after_sim.nii.gz", mask_after, affine)
    (out_dir / "sim_info.json").write_text(json.dumps(sim_info, indent=2), encoding="utf-8")

    # AFTER projections
    projs_after = make_three_projections(mask_after)
    for k, img in projs_after.items():
        cv2.imwrite(str(out_dir / f"proj_{k}_after.png"), img)

    # montage before/after
    save_montage_before_after(out_dir / "montage_before_after.png", projs_before, projs_after)

    # YOLO on AFTER projections
    model = YOLO(str(weights))
    class_names = load_class_names(det_dir, model)

    detections_all: Dict[str, List[Dict]] = {}
    det_imgs: Dict[str, np.ndarray] = {}

    for k in ["axial", "coronal", "sagittal"]:
        img_bgr = to_3ch(projs_after[k])
        dets = run_yolo_on_image(model, img_bgr, conf=args.conf, iou=args.iou, device=args.device)
        detections_all[k] = dets

        vis = draw_detections(img_bgr, dets, class_names)
        det_imgs[k] = vis
        cv2.imwrite(str(out_dir / f"det_{k}.png"), vis)

    save_montage_yolo(out_dir / "montage_yolo.png", projs_after, det_imgs)

    # dump detections
    payload = {
        "mask": str(mask_path),
        "pixdim_xyz": list(pixdim),
        "simulate": bool(args.simulate),
        "sim_info": sim_info,
        "yolo": {
            "weights": str(weights),
            "conf": float(args.conf),
            "iou": float(args.iou),
            "device": args.device,
            "class_names": {str(k): v for k, v in class_names.items()},
        },
        "detections": detections_all,
    }
    (out_dir / "detections.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("OK")
    print("Saved:")
    print(" - proj_*_before.png / proj_*_after.png")
    print(" - montage_before_after.png")
    print(" - det_*.png")
    print(" - montage_yolo.png")
    print(" - mask_after_sim.nii.gz")
    print(" - sim_info.json, detections.json")


if __name__ == "__main__":
    main()


