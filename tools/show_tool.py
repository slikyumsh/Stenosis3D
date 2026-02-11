import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

MANIFEST = "data/manifest.csv"

df = pd.read_csv(MANIFEST)

# случайный кейс из манифеста (без зависимости от split)
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

# нормализация для просмотра
lo, hi = np.percentile(img, (1, 99))
imgv = np.clip(img, lo, hi)

# выбираем срезы так, чтобы маска была видна:
# sagittal: фиксируем x -> суммируем маску по (y,z)
x_idx = int(np.argmax((msk > 0).sum(axis=(1, 2))))
# coronal: фиксируем y -> суммируем по (x,z)
y_idx = int(np.argmax((msk > 0).sum(axis=(0, 2))))
# axial: фиксируем z -> суммируем по (x,y)
z_idx = int(np.argmax((msk > 0).sum(axis=(0, 1))))

print(f"Best slices: x={x_idx}, y={y_idx}, z={z_idx}")
vals = np.unique(msk)
print("Mask unique values (first 10):", vals[:10], " ... total:", len(vals))

plt.figure(figsize=(14, 4))

# Sagittal (x fixed): (y,z)
plt.subplot(1, 3, 1)
plt.imshow(imgv[x_idx, :, :].T, cmap="gray", origin="lower")
plt.contour((msk[x_idx, :, :] > 0).T, levels=[0.5], origin="lower")
plt.title(f"Sagittal x={x_idx}")
plt.axis("off")

# Coronal (y fixed): (x,z)
plt.subplot(1, 3, 2)
plt.imshow(imgv[:, y_idx, :].T, cmap="gray", origin="lower")
plt.contour((msk[:, y_idx, :] > 0).T, levels=[0.5], origin="lower")
plt.title(f"Coronal y={y_idx}")
plt.axis("off")

# Axial (z fixed): (x,y)
plt.subplot(1, 3, 3)
plt.imshow(imgv[:, :, z_idx].T, cmap="gray", origin="lower")
plt.contour((msk[:, :, z_idx] > 0).T, levels=[0.5], origin="lower")
plt.title(f"Axial z={z_idx}")
plt.axis("off")

plt.tight_layout()
plt.show()
