from pathlib import Path
import math
import cv2
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm
from radiomics import imageoperations

PROJECT_ROOT = Path.cwd()

ROI_XAI_MANIFEST = Path(
    "data/processed/cbis_ddsm/manifests/cbis_ddsm_roi_xai_manifest.csv"
)

OUTPUT_DIR = Path("results/radiomics/discretisation_pilot")
OUTPUT_CSV = OUTPUT_DIR / "radiomics_discretisation_pilot.csv"
SUMMARY_TXT = OUTPUT_DIR / "radiomics_discretisation_summary.txt"

RANDOM_SEED = 42
TARGET_SAMPLE_SIZE = 96
NORMALIZE_SCALE = 100.0
BIN_WIDTHS = [2.0, 5.0, 10.0]


def resolve_path(value):
    p = Path(str(value))
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def to_uint8(array):
    array = np.asarray(array)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = float(array.min()), float(array.max())
    if mx <= mn:
        return np.zeros(array.shape, dtype=np.uint8)
    out = (array.astype(np.float32) - mn) / (mx - mn) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def corner_background_is_white(binary):
    h, w = binary.shape
    ph = max(1, int(round(h * 0.05)))
    pw = max(1, int(round(w * 0.05)))
    patches = [
        binary[:ph, :pw],
        binary[:ph, w-pw:],
        binary[h-ph:, :pw],
        binary[h-ph:, w-pw:],
    ]
    return np.mean([p.mean() for p in patches]) > 127.5


def binarise_roi_mask(array):
    image = to_uint8(array)
    if image.max() == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    _, thresholded = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if corner_background_is_white(thresholded):
        thresholded = 255 - thresholded
    return (thresholded > 0).astype(np.uint8)


def load_dicom_pixels(path):
    ds = pydicom.dcmread(str(path))
    pixels = np.squeeze(np.asarray(ds.pixel_array))
    if pixels.ndim != 2:
        raise ValueError(f"Expected 2-D DICOM image, got {pixels.shape}")
    return pixels


def match_mask_to_image(mask, image_shape):
    if mask.shape == image_shape:
        return mask, False
    h, w = image_shape
    resized = cv2.resize(
        mask.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )
    return (resized > 0).astype(np.uint8), True


def normalize_like_pyradiomics(image, scale):
    image = image.astype(np.float64)
    mean = float(image.mean())
    std = float(image.std())
    if std <= 0:
        raise ValueError("Image standard deviation is zero.")
    return scale * (image - mean) / std


def discretisation_stats(normalized_image, mask, bin_width):
    coords = np.where(mask > 0)
    discretized, _ = imageoperations.binImage(
        normalized_image,
        parameterMatrixCoordinates=coords,
        binWidth=float(bin_width),
    )
    roi_bins = discretized[coords]
    return int(np.unique(roi_bins).size)


def build_stratified_sample(frame, target_size):
    groups = list(frame.groupby(
        ["abnormality type", "pathology"],
        dropna=False,
        observed=True,
    ))
    per_group = max(1, math.ceil(target_size / len(groups)))
    parts = []
    for _, group in groups:
        n = min(per_group, len(group))
        parts.append(group.sample(n=n, random_state=RANDOM_SEED))
    sample = pd.concat(parts, ignore_index=True)
    if len(sample) > target_size:
        sample = sample.sample(n=target_size, random_state=RANDOM_SEED)
    return sample.reset_index(drop=True)


if not ROI_XAI_MANIFEST.exists():
    raise FileNotFoundError(ROI_XAI_MANIFEST.resolve())

roi_df = pd.read_csv(ROI_XAI_MANIFEST)
usable = roi_df[roi_df["roi_xai_status"] == "success"].copy()

sample = build_stratified_sample(usable, TARGET_SAMPLE_SIZE)

print("=" * 72)
print("PYRADIOMICS DISCRETISATION PILOT")
print("=" * 72)
print(f"\nUsable ROI rows: {len(usable):,}")
print(f"Pilot sample size: {len(sample):,}")
print("\nPilot strata:")
print(sample.groupby(
    ["abnormality type", "pathology"],
    observed=True,
).size())

records = []

for _, row in tqdm(
    sample.iterrows(),
    total=len(sample),
    desc="Testing discretisation",
):
    rec = {
        "patient_id": row["patient_id"],
        "abnormality_type": row["abnormality type"],
        "pathology": row["pathology"],
        "abnormality_id": row.get("abnormality id", np.nan),
    }

    try:
        image = load_dicom_pixels(resolve_path(row["full_mammogram_path"]))
        roi = load_dicom_pixels(resolve_path(row["roi_mask_path"]))
        mask = binarise_roi_mask(roi)
        mask, resized = match_mask_to_image(mask, image.shape)

        if np.count_nonzero(mask) == 0:
            raise ValueError("ROI has no foreground.")

        normalized = normalize_like_pyradiomics(
            image,
            NORMALIZE_SCALE,
        )

        roi_values = normalized[mask > 0]

        rec.update({
            "status": "success",
            "roi_pixel_count": int(np.count_nonzero(mask)),
            "roi_fraction_of_image": float(np.mean(mask > 0)),
            "roi_resized_to_full_image": bool(resized),
            "normalized_roi_min": float(roi_values.min()),
            "normalized_roi_max": float(roi_values.max()),
            "normalized_roi_mean": float(roi_values.mean()),
            "normalized_roi_std": float(roi_values.std()),
        })

        for width in BIN_WIDTHS:
            suffix = str(width).replace(".", "_")
            rec[f"binwidth_{suffix}_n_bins"] = discretisation_stats(
                normalized,
                mask,
                width,
            )

    except Exception as exc:
        rec["status"] = f"error: {exc}"

    records.append(rec)

result = pd.DataFrame(records)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_CSV, index=False)

success = result[result["status"] == "success"].copy()

lines = []
def emit(text=""):
    print(text)
    lines.append(str(text))

emit("\n" + "=" * 72)
emit("DISCRETISATION PILOT SUMMARY")
emit("=" * 72)
emit(f"\nSuccessful rows: {len(success):,} / {len(result):,}")

emit("\nNormalized ROI intensity summary:")
emit(success[
    ["normalized_roi_min", "normalized_roi_max",
    "normalized_roi_mean", "normalized_roi_std"]
].describe().to_string())

candidate_rows = []

for width in BIN_WIDTHS:
    suffix = str(width).replace(".", "_")
    col = f"binwidth_{suffix}_n_bins"

    emit(f"\nBin width = {width:g}")
    emit(success[col].describe().to_string())

    n_lt16 = int((success[col] < 16).sum())
    n_16_29 = int(((success[col] >= 16) & (success[col] < 30)).sum())
    n_30_130 = int(((success[col] >= 30) & (success[col] <= 130)).sum())
    n_gt130 = int((success[col] > 130).sum())

    emit(f"<16 bins: {n_lt16}")
    emit(f"16-29 bins: {n_16_29}")
    emit(f"30-130 bins: {n_30_130}")
    emit(f">130 bins: {n_gt130}")

    candidate_rows.append({
        "bin_width": width,
        "median_bins": float(success[col].median()),
        "mean_bins": float(success[col].mean()),
        "fraction_30_130": float(
            ((success[col] >= 30) & (success[col] <= 130)).mean()
        ),
    })

emit("\nBin counts by lesion type:")
for width in BIN_WIDTHS:
    suffix = str(width).replace(".", "_")
    col = f"binwidth_{suffix}_n_bins"
    emit(f"\nBin width = {width:g}")
    emit(success.groupby("abnormality_type", observed=True)[col]
        .agg(["count", "mean", "median", "min", "max"])
        .to_string())

emit("\nCandidate comparison:")
candidate_df = pd.DataFrame(candidate_rows).sort_values(
    ["fraction_30_130", "bin_width"],
    ascending=[False, True],
)
emit(candidate_df.to_string(index=False))

if len(candidate_df):
    emit(
        f"\nBest pilot candidate by proportion in the 30-130 bin range: "
        f"binWidth = {candidate_df.iloc[0]['bin_width']:g}"
    )

SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")

emit("\nSaved CSV:")
emit(OUTPUT_CSV.resolve())
emit("\nSaved summary:")
emit(SUMMARY_TXT.resolve())
