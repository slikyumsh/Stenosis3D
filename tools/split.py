"""
Prepare train/test folders and a manifest.csv from raw NIfTI pairs.

- Finds all *.img.nii.gz files under RAW_ROOT and matches them with *.label.nii.gz.
- Splits cases into train/test.
- Copies or moves files into data/train and data/test using {case_id}.nii.gz naming.
- Writes data/manifest.csv with updated paths and split column.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# ------------------- settings -------------------
TEST_SIZE = 0.20
SEED = 42

# MODE: "copy" (recommended) or "move"
MODE = "copy"
# ------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RAW_ROOT = PROJECT_ROOT / "raw_data"
OUT_ROOT = PROJECT_ROOT / "data"

TRAIN_IMG_DIR = OUT_ROOT / "train" / "images"
TRAIN_MSK_DIR = OUT_ROOT / "train" / "masks"
TEST_IMG_DIR = OUT_ROOT / "test" / "images"
TEST_MSK_DIR = OUT_ROOT / "test" / "masks"


def norm_id(x):
    """Normalize case_id strings (e.g., '1.0' -> '1')."""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def ensure_dirs():
    """Create output directories if they do not exist."""
    for p in [TRAIN_IMG_DIR, TRAIN_MSK_DIR, TEST_IMG_DIR, TEST_MSK_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def transfer(src: Path, dst: Path):
    """Copy or move a file depending on MODE (skips if dst already exists)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return  # already created
    if MODE == "move":
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


print("PROJECT_ROOT:", PROJECT_ROOT)
print("RAW_ROOT    :", RAW_ROOT.resolve())
print("OUT_ROOT    :", OUT_ROOT.resolve())
print("MODE        :", MODE)

# --- 1) Collect image/label pairs ---
img_files = list(RAW_ROOT.rglob("*.img.nii.gz"))
if not img_files:
    raise SystemExit(f"[ERROR] No *.img.nii.gz found in {RAW_ROOT.resolve()}")

rows = []
missing_labels = 0
for img_path in img_files:
    stem = img_path.name.replace(".img.nii.gz", "")  # e.g. "1"
    label_path = img_path.with_name(f"{stem}.label.nii.gz")
    if label_path.exists():
        rows.append(
            {
                "case_id": norm_id(stem),
                "image_path": str(img_path),
                "mask_path": str(label_path),
            }
        )
    else:
        missing_labels += 1

if not rows:
    raise SystemExit("[ERROR] No img+label pairs were found. Check file naming.")

manifest = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
print(f"Pairs found: {len(manifest)} | imgs without labels: {missing_labels}")

# --- 2) Train/test split ---
case_ids = manifest["case_id"].tolist()
train_ids, test_ids = train_test_split(case_ids, test_size=TEST_SIZE, random_state=SEED, shuffle=True)

train_set = set(train_ids)
manifest["split"] = manifest["case_id"].apply(lambda x: "train" if x in train_set else "test")

print("\nSplit counts:")
print(manifest["split"].value_counts())

# --- 3) Copy/move into data/train and data/test ---
ensure_dirs()

# Save as standard names: {case_id}.nii.gz (simplifies downstream loaders)
for i, r in manifest.iterrows():
    cid = r["case_id"]
    img_src = Path(r["image_path"])
    msk_src = Path(r["mask_path"])

    if r["split"] == "train":
        img_dst = TRAIN_IMG_DIR / f"{cid}.nii.gz"
        msk_dst = TRAIN_MSK_DIR / f"{cid}.nii.gz"
    else:
        img_dst = TEST_IMG_DIR / f"{cid}.nii.gz"
        msk_dst = TEST_MSK_DIR / f"{cid}.nii.gz"

    transfer(img_src, img_dst)
    transfer(msk_src, msk_dst)

    # Light progress logging
    if (i + 1) % 100 == 0:
        print(f"Processed {i+1}/{len(manifest)}")

# --- 4) Update manifest paths to the new locations ---
def new_paths(row):
    """Return updated (image_path, mask_path) for a manifest row based on split."""
    cid = row["case_id"]
    if row["split"] == "train":
        return pd.Series(
            {
                "image_path": str(TRAIN_IMG_DIR / f"{cid}.nii.gz"),
                "mask_path": str(TRAIN_MSK_DIR / f"{cid}.nii.gz"),
            }
        )
    else:
        return pd.Series(
            {
                "image_path": str(TEST_IMG_DIR / f"{cid}.nii.gz"),
                "mask_path": str(TEST_MSK_DIR / f"{cid}.nii.gz"),
            }
        )


manifest[["image_path", "mask_path"]] = manifest.apply(new_paths, axis=1)

# --- 5) Save final manifest ---
out_csv = OUT_ROOT / "manifest.csv"
manifest.to_csv(out_csv, index=False)
print("\nSaved manifest:", out_csv.resolve())
print("Train images:", len(list(TRAIN_IMG_DIR.glob("*.nii.gz"))))
print("Test images :", len(list(TEST_IMG_DIR.glob("*.nii.gz"))))
