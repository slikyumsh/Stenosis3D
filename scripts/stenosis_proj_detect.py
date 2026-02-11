from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt

try:
    import yaml
except Exception:
    yaml = None

from ultralytics import YOLO


# ----------------------------- IO ----------------------------- #
def load_mask_nii(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Load NIfTI mask and return (mask_u8, pixdim_xyz).
    mask_u8: uint8 array with values 0/1
    """
    nii = nib.load(str(path))
    data = nii.get_fdata(dtype=np.float32)
    mask = (data > 0).astype(np.uint8)

    zooms = nii.header.get_zooms()
    pixdim = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return mask, pixdim


# ----------------------------- Projections ----------------------------- #
def mip_projection(mask_u8: np.ndarray, axis: int) -> np.ndarray:
    """
    Maximum Intensity Projection for binary mask -> 2D uint8 {0,255}.
    axis:
      0 -> project along X => YZ (sagittal view)
      1 -> project along Y => XZ (coronal view)
      2 -> project along Z => XY (axial view)
    """
    proj = np.max(mask_u8, axis=axis).astype(np.uint8)  # 0/1
    proj = (proj * 255).astype(np.uint8)
    return proj


def make_three_projections(mask_u8: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Returns dict with keys:
      axial    : XY (axis=2)
      coronal  : XZ (axis=1)
      sagittal : YZ (axis=0)
    """
    return {
        "axial": mip_projection(mask_u8, axis=2),
        "coronal": mip_projection(mask_u8, axis=1),
        "sagittal": mip_projection(mask_u8, axis=0),
    }


def to_3ch(img_u8: np.ndarray) -> np.ndarray:
    """Convert 1ch uint8 -> 3ch BGR uint8 for YOLO/OpenCV."""
    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)


# ----------------------------- Detector helpers ----------------------------- #
def load_class_names(detector_dir: Path, model: YOLO) -> Dict[int, str]:
    """
    Try to load class names:
      1) from model.names (preferred)
      2) from detector/data.yaml (fallback)
    """
    names = None
    try:
        names = model.names
    except Exception:
        names = None

    if isinstance(names, dict) and len(names) > 0:
        return {int(k): str(v) for k, v in names.items()}

    # fallback: data.yaml
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

    return {0: "stenosis"}  # last resort


def run_yolo_on_image(
    model: YOLO,
    img_bgr: np.ndarray,
    conf: float,
    iou: float,
    device: str,
) -> List[Dict]:
    """
    Returns list of detections with:
      - xyxy (x1,y1,x2,y2)
      - conf
      - cls
    """
    # ultralytics accepts numpy arrays directly
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


def draw_detections(
    img_bgr: np.ndarray,
    dets: List[Dict],
    class_names: Dict[int, str],
    color: Tuple[int, int, int] = (0, 0, 255),  # red in BGR
    thickness: int = 2,
) -> np.ndarray:
    out = img_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        conf = d["conf"]
        cls = d["cls"]
        label = f"{class_names.get(cls, str(cls))} {conf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        # label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1, y_text), (x1 + tw + 4, y_text + th + 6), color, -1)
        cv2.putText(out, label, (x1 + 2, y_text + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ----------------------------- Visualization ----------------------------- #
def save_montage(
    out_png: Path,
    projs_gray: Dict[str, np.ndarray],
    det_bgr: Dict[str, np.ndarray],
):
    """
    2 rows x 3 cols:
      row0: projections
      row1: detections
    """
    keys = ["axial", "coronal", "sagittal"]

    plt.figure(figsize=(16, 9), dpi=160)

    for i, k in enumerate(keys):
        # projection
        plt.subplot(2, 3, i + 1)
        plt.imshow(projs_gray[k], cmap="gray")
        plt.title(f"{k} projection")
        plt.axis("off")

        # detection
        plt.subplot(2, 3, i + 4)
        # OpenCV BGR -> RGB for matplotlib
        plt.imshow(cv2.cvtColor(det_bgr[k], cv2.COLOR_BGR2RGB))
        plt.title(f"{k} + YOLO boxes")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# ----------------------------- Main ----------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", type=str, required=True, help="Path to 3D mask NIfTI (.nii/.nii.gz).")
    ap.add_argument("--detector-dir", type=str, default="detector", help="Folder with best.pt and data.yaml.")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory for projections/detections.")
    ap.add_argument("--conf", type=float, default=0.001, help="YOLO confidence threshold.")
    ap.add_argument("--iou", type=float, default=0.001, help="YOLO NMS IoU threshold.")
    ap.add_argument("--device", type=str, default="",
                    help="YOLO device: ''=auto, 'cpu', '0' (GPU0), '1' (GPU1), etc.")
    args = ap.parse_args()

    mask_path = Path(args.mask)
    det_dir = Path(args.detector_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = det_dir / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Detector weights not found: {weights}")

    # load mask
    mask_u8, pixdim = load_mask_nii(mask_path)

    # projections
    projs = make_three_projections(mask_u8)

    # save raw projections
    proj_paths = {}
    for k, img in projs.items():
        p = out_dir / f"proj_{k}.png"
        cv2.imwrite(str(p), img)  # grayscale png
        proj_paths[k] = str(p)

    # load model
    model = YOLO(str(weights))
    class_names = load_class_names(det_dir, model)

    # device for ultralytics
    device = args.device  # '' means auto in ultralytics

    # run detection on each projection
    detections_all: Dict[str, List[Dict]] = {}
    det_imgs: Dict[str, np.ndarray] = {}

    for k in ["axial", "coronal", "sagittal"]:
        img_gray = projs[k]
        img_bgr = to_3ch(img_gray)

        dets = run_yolo_on_image(model, img_bgr, conf=args.conf, iou=args.iou, device=device)
        detections_all[k] = dets

        img_vis = draw_detections(img_bgr, dets, class_names, color=(0, 0, 255), thickness=2)
        det_imgs[k] = img_vis

        out_det = out_dir / f"det_{k}.png"
        cv2.imwrite(str(out_det), img_vis)

    # montage
    montage_path = out_dir / "montage.png"
    save_montage(montage_path, projs, det_imgs)

    # save json with detections + metadata
    out_json = out_dir / "detections.json"
    payload = {
        "mask": str(mask_path),
        "pixdim_xyz": list(pixdim),
        "projections": proj_paths,
        "conf": float(args.conf),
        "iou": float(args.iou),
        "detections": detections_all,
        "class_names": {str(k): v for k, v in class_names.items()},
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("OK")
    print("Saved projections :", [f"proj_{k}.png" for k in ["axial", "coronal", "sagittal"]])
    print("Saved detections  :", [f"det_{k}.png" for k in ["axial", "coronal", "sagittal"]])
    print("Saved montage     :", montage_path.resolve())
    print("Saved json        :", out_json.resolve())


if __name__ == "__main__":
    main()


