from pathlib import Path
import math

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


MANIFEST = Path(
    "data/processed/cbis_ddsm/manifests/"
    "cbis_ddsm_full_image_manifest_preprocessed.csv"
)

OUTPUT_DIR = Path("results/figures/preprocessing_qa")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SAMPLES_PER_GROUP = 2


def load_gray(path):
    with Image.open(path) as img:
        return np.asarray(img.convert("L"))


def resolve_path(value):
    path = Path(str(value))
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def orientation_score(image):
    """
    Positive values mean more foreground is present near the left
    edge than the right edge. This is a simple QA check only.
    """
    _, mask = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    width = mask.shape[1]
    edge = max(1, int(width * 0.20))
    left = np.count_nonzero(mask[:, :edge])
    right = np.count_nonzero(mask[:, -edge:])
    total = max(1, left + right)
    return (left - right) / total


df = pd.read_csv(MANIFEST)
success = df[df["preprocessing_status"] == "success"].copy()

print("=" * 72)
print("PREPROCESSED IMAGE QA")
print("=" * 72)
print(f"\nManifest rows: {len(df):,}")
print(f"Successful rows: {len(success):,}")


# ----------------------------------------------------------------
# File and pixel checks
# ----------------------------------------------------------------
records = []

for _, row in success.iterrows():
    path = resolve_path(row["preprocessed_path"])

    record = {
        "image_id": row["image_id"],
        "patient_id": row["patient_id"],
        "split": row["split"],
        "path": str(path),
        "exists": path.exists(),
    }

    if not path.exists():
        records.append(record)
        continue

    image = load_gray(path)

    record.update({
        "width": image.shape[1],
        "height": image.shape[0],
        "min_pixel": int(image.min()),
        "max_pixel": int(image.max()),
        "mean_pixel": float(image.mean()),
        "std_pixel": float(image.std()),
        "nonzero_fraction": float(np.count_nonzero(image) / image.size),
        "orientation_score": float(orientation_score(image)),
    })

    records.append(record)

qa = pd.DataFrame(records)
qa.to_csv(
    "data/processed/cbis_ddsm/manifests/"
    "cbis_ddsm_full_image_preprocessing_qa.csv",
    index=False
)

print("\nFiles found:")
print(qa["exists"].value_counts(dropna=False))

valid = qa[qa["exists"] == True]

print("\nImage sizes:")
print(valid[["width", "height"]].value_counts().head())

print("\nBlank / near-blank images (std < 1):")
print((valid["std_pixel"] < 1).sum())

print("\nNon-zero fraction:")
print(valid["nonzero_fraction"].describe())

print("\nOrientation score:")
print(valid["orientation_score"].describe())

print("\nImages with more edge foreground on the RIGHT after normalisation:")
print((valid["orientation_score"] < 0).sum())


# ----------------------------------------------------------------
# Stratified visual sample
# ----------------------------------------------------------------
def lesion_type(row):
    if bool(row.get("contains_mass", False)):
        return "mass"
    if bool(row.get("contains_calcification", False)):
        return "calcification"
    return "unknown"


success["lesion_type"] = success.apply(lesion_type, axis=1)

group_columns = [
    "binary_pathology",
    "lesion_type",
    "image_view",
]

samples = []

for key, group in success.groupby(group_columns):
    n = min(SAMPLES_PER_GROUP, len(group))
    sampled = group.sample(n=n, random_state=RANDOM_STATE)
    samples.append(sampled)

sample_df = pd.concat(samples, ignore_index=True)

n = len(sample_df)
cols = 4
rows = math.ceil(n / cols)

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(12, 3.5 * rows)
)

axes = np.array(axes).reshape(-1)

for ax in axes:
    ax.axis("off")

for ax, (_, row) in zip(axes, sample_df.iterrows()):
    path = resolve_path(row["preprocessed_path"])
    image = load_gray(path)

    ax.imshow(image, cmap="gray")
    ax.axis("off")
    ax.set_title(
        f"{row['patient_id']} | {row['binary_pathology']}\n"
        f"{row['lesion_type']} | {row['image_view']} | {row['split']}",
        fontsize=8
    )

fig.suptitle(
    "CBIS-DDSM Preprocessed Full-Mammogram Visual QA",
    fontsize=14
)

fig.tight_layout()
output_figure = OUTPUT_DIR / "preprocessing_stratified_sample.png"
fig.savefig(output_figure, dpi=180, bbox_inches="tight")
plt.close(fig)

print("\nSaved QA manifest:")
print(
    Path(
        "data/processed/cbis_ddsm/manifests/"
        "cbis_ddsm_full_image_preprocessing_qa.csv"
    ).resolve()
)

print("\nSaved visual QA figure:")
print(output_figure.resolve())
