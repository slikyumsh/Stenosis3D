"""
Quick 3-view visualization of a random case from a segmentation manifest.

Loads a random image/mask pair from MANIFEST, checks shape consistency, picks
the slices where the mask is most visible (max projected mask area), and shows
sagittal/coronal/axial views with mask contours overlaid.
"""

import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

MANIFEST = "data/manifest.csv"

df = pd.read_csv(MANIFEST)

# Pick a random case from the manifest (independent of split)
row = df.sample(1, random_state=None).iloc[0]

case_id = row["case_id"]
img_path = row["image_path"]
msk_path = row["mask_path"]

print("CASE:", case_id)
print("IMG :", img_path)
print("MSK :", msk_path)

img = nib.load(img_path).get_fdata(dtype=np.float32)
msk = nib.load(msk_path).get_fdata(dtype=np.float32)

if img.shape != msk.shape:
    raise ValueError(f"Shape mismatch: img={img.shape}, mask={msk.shape}")

# Normalize intensities for display
lo, hi = np.percentile(img, (1, 99))
imgv = np.clip(img, lo, hi)

# Pick slices where the mask is most visible:
# - sagittal: fix x, maximize mask sum over (y, z)
# - coronal:  fix y, maximize mask sum over (x, z)
# - axial:    fix z, maximize mask sum over (x, y)
mask_bin = (msk > 0)
x_idx = int(np.argmax(mask_bin.sum(axis=(1, 2))))
y_idx = int(np.argmax(mask_bin.sum(axis=(0, 2))))
z_idx = int(np.argmax(mask_bin.sum(axis=(0, 1))))

print(f"Best slices: x={x_idx}, y={y_idx}, z={z_idx}")
vals = np.unique(msk)
print("Mask unique values (first 10):", vals[:10], " ... total:", len(vals))

plt.figure(figsize=(14, 4))

# Sagittal (x fixed): (y, z)
plt.subplot(1, 3, 1)
plt.imshow(imgv[x_idx, :, :].T, cmap="gray", origin="lower")
plt.contour(mask_bin[x_idx, :, :].T, levels=[0.5], origin="lower")
plt.title(f"Sagittal x={x_idx}")
plt.axis("off")

# Coronal (y fixed): (x, z)
plt.subplot(1, 3, 2)
plt.imshow(imgv[:, y_idx, :].T, cmap="gray", origin="lower")
plt.contour(mask_bin[:, y_idx, :].T, levels=[0.5], origin="lower")
plt.title(f"Coronal y={y_idx}")
plt.axis("off")

# Axial (z fixed): (x, y)
plt.subplot(1, 3, 3)
plt.imshow(imgv[:, :, z_idx].T, cmap="gray", origin="lower")
plt.contour(mask_bin[:, :, z_idx].T, levels=[0.5], origin="lower")
plt.title(f"Axial z={z_idx}")
plt.axis("off")

plt.tight_layout()
plt.show()
