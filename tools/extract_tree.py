from pathlib import Path
import numpy as np
import nibabel as nib

# --- настройки ---
MANIFEST = Path("data/manifest.csv")   # или prepared_manifest.csv
CASE_ID = "100"                          # <- поменяй на нужный
OUT_DIR = Path("exports")              # куда сохранять
EXPORT_MESH = True                     # False если нужен только NIfTI
KEEP_LARGEST_COMPONENT = True          # убрать мелкий шум
# -----------------

def keep_largest_cc(mask: np.ndarray) -> np.ndarray:
    """Оставляет только крупнейшую связную компоненту (3D)."""
    from collections import deque

    mask = (mask > 0).astype(np.uint8)
    visited = np.zeros(mask.shape, dtype=np.uint8)

    # 6-связность
    neigh = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

    best_count = 0
    best_voxels = None

    it = np.argwhere(mask > 0)
    for sx, sy, sz in it:
        if visited[sx, sy, sz]:
            continue
        q = deque([(sx, sy, sz)])
        visited[sx, sy, sz] = 1
        vox = [(sx, sy, sz)]
        while q:
            x, y, z = q.popleft()
            for dx, dy, dz in neigh:
                nx, ny, nz = x + dx, y + dy, z + dz
                if (0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1] and 0 <= nz < mask.shape[2]):
                    if mask[nx, ny, nz] and not visited[nx, ny, nz]:
                        visited[nx, ny, nz] = 1
                        q.append((nx, ny, nz))
                        vox.append((nx, ny, nz))
        if len(vox) > best_count:
            best_count = len(vox)
            best_voxels = vox

    out = np.zeros_like(mask, dtype=np.uint8)
    if best_voxels is not None:
        xs, ys, zs = zip(*best_voxels)
        out[np.array(xs), np.array(ys), np.array(zs)] = 1
    return out

def main():
    import pandas as pd

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST)
    row = df[df["case_id"].astype(str) == str(CASE_ID)].iloc[0]

    img_path = Path(row["image_path"])
    msk_path = Path(row["mask_path"])

    print("CASE:", CASE_ID)
    print("IMG :", img_path)
    print("MSK :", msk_path)

    # читаем маску и сохраняем affine/spacing из исходного NIfTI
    msk_nii = nib.load(str(msk_path))
    msk = msk_nii.get_fdata(dtype=np.float32)

    mask_bin = (msk > 0).astype(np.uint8)

    if KEEP_LARGEST_COMPONENT:
        print("Keeping largest connected component...")
        mask_bin = keep_largest_cc(mask_bin)

    # 1) сохраняем mask-only volume
    out_mask_path = OUT_DIR / f"{CASE_ID}_mask_only.nii.gz"
    out_nii = nib.Nifti1Image(mask_bin, affine=msk_nii.affine, header=msk_nii.header)
    out_nii.set_data_dtype(np.uint8)
    nib.save(out_nii, str(out_mask_path))
    print("Saved mask-only NIfTI:", out_mask_path.resolve())

    if not EXPORT_MESH:
        return

    # 2) экспорт mesh через marching cubes
    print("Exporting mesh (marching cubes)...")
    try:
        from skimage import measure
    except ImportError:
        raise SystemExit("Нужно поставить scikit-image: pip install scikit-image")

    # spacing берём из header (pixdim)
    zooms = msk_nii.header.get_zooms()[:3]  # (sx, sy, sz)
    verts, faces, _, _ = measure.marching_cubes(mask_bin, level=0.5, spacing=zooms)

    # сохраним в PLY (простой ASCII)
    out_ply = OUT_DIR / f"{CASE_ID}_artery_tree.ply"
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

    print("Saved mesh PLY:", out_ply.resolve())
    print("\nОткрытие:")
    print("- mask-only NIfTI: 3D Slicer / napari (Volume Rendering)")
    print("- mesh PLY: MeshLab / Blender / 3D Slicer")

if __name__ == "__main__":
    main()
