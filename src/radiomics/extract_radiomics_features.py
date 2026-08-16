from pathlib import Path
import json
import time

import cv2
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from tqdm import tqdm
from radiomics import featureextractor

PROJECT_ROOT = Path.cwd()
ROI_XAI_MANIFEST = Path(
    "data/processed/cbis_ddsm/manifests/cbis_ddsm_roi_xai_manifest.csv"
)
PARAMS_FILE = Path("src/radiomics/radiomics_params.yaml")
OUTPUT_DIR = Path("data/processed/cbis_ddsm/radiomics")
OUTPUT_CSV = OUTPUT_DIR / "cbis_ddsm_radiomics_lesion_features.csv"
ERROR_CSV = OUTPUT_DIR / "cbis_ddsm_radiomics_extraction_errors.csv"
SUMMARY_JSON = OUTPUT_DIR / "cbis_ddsm_radiomics_extraction_summary.json"
CHECKPOINT_EVERY = 100
MAX_CASES = None  # Set to 20 for a smoke test; leave None for full extraction.


def resolve_path(value):
    if pd.isna(value):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_dicom_pixels(path):
    ds = pydicom.dcmread(str(path))
    array = np.squeeze(np.asarray(ds.pixel_array))
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D DICOM image, got {array.shape}.")
    return array


def to_uint8(array):
    array = np.asarray(array)
    if array.size == 0:
        raise ValueError("ROI array is empty.")
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum <= minimum:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (array.astype(np.float32) - minimum) / (maximum - minimum) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def corner_background_is_white(binary_255):
    h, w = binary_255.shape
    ph = max(1, int(round(h * 0.05)))
    pw = max(1, int(round(w * 0.05)))
    patches = [
        binary_255[:ph, :pw],
        binary_255[:ph, w - pw:],
        binary_255[h - ph:, :pw],
        binary_255[h - ph:, w - pw:],
    ]
    return np.mean([float(p.mean()) for p in patches]) > 127.5


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


def align_mask_to_full_image(mask, image_shape):
    if mask.shape == image_shape:
        return mask, False
    height, width = image_shape
    resized = cv2.resize(
        mask.astype(np.uint8),
        (int(width), int(height)),
        interpolation=cv2.INTER_NEAREST,
    )
    return (resized > 0).astype(np.uint8), True


def numpy_to_sitk_image(image_array, mask_array):
    image = sitk.GetImageFromArray(image_array.astype(np.float32))
    mask = sitk.GetImageFromArray(mask_array.astype(np.uint8))

    # No reliable physical pixel spacing is available in these CBIS-DDSM
    # Secondary Capture files, so extraction is performed in pixel units.
    for obj in (image, mask):
        obj.SetSpacing((1.0, 1.0))
        obj.SetOrigin((0.0, 0.0))
        obj.SetDirection((1.0, 0.0, 0.0, 1.0))

    return image, mask


def scalar_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        array = np.asarray(value)
        if array.size == 1:
            return array.item()
    except Exception:
        pass
    return str(value)


def extract_feature_columns(result):
    # Exclude diagnostics/provenance from model predictors.
    return {
        key: scalar_value(value)
        for key, value in result.items()
        if not key.startswith("diagnostics_")
    }


def write_checkpoint(records):
    if not records:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)


if not ROI_XAI_MANIFEST.exists():
    raise FileNotFoundError(f"ROI manifest not found: {ROI_XAI_MANIFEST.resolve()}")
if not PARAMS_FILE.exists():
    raise FileNotFoundError(f"Parameter file not found: {PARAMS_FILE.resolve()}")

manifest = pd.read_csv(ROI_XAI_MANIFEST)
required_columns = [
    "roi_xai_status",
    "patient_id",
    "full_image_id",
    "dataset",
    "full_mammogram_path",
    "roi_mask_path",
    "abnormality id",
    "abnormality type",
    "pathology",
]
missing = [c for c in required_columns if c not in manifest.columns]
if missing:
    raise ValueError("ROI manifest is missing required columns: " + ", ".join(missing))

usable = manifest[manifest["roi_xai_status"] == "success"].copy().reset_index(drop=True)
if MAX_CASES is not None:
    usable = usable.head(int(MAX_CASES)).copy()

extractor = featureextractor.RadiomicsFeatureExtractor(str(PARAMS_FILE))

print("=" * 72)
print("CBIS-DDSM PYRADIOMICS FEATURE EXTRACTION")
print("=" * 72)
print(f"\nParameter file: {PARAMS_FILE.resolve()}")
print(f"Usable lesion ROIs: {len(usable):,}")
print("\nLesion type:")
print(usable["abnormality type"].value_counts())
print("\nPathology:")
print(usable["pathology"].value_counts())
print("\nEnabled image types:")
print(extractor.enabledImagetypes)
print("\nEnabled feature classes:")
print(list(extractor.enabledFeatures.keys()))
print("\nSettings:")
for key in ["force2D", "normalize", "normalizeScale", "binWidth"]:
    print(f"{key}: {extractor.settings.get(key)}")

records = []
errors = []
start_time = time.time()

for index, row in tqdm(
    usable.iterrows(),
    total=len(usable),
    desc="Extracting radiomics",
):
    metadata = {
        "source_manifest_index": int(index),
        "full_image_id": row["full_image_id"],
        "patient_id": row["patient_id"],
        "dataset": row["dataset"],
        "abnormality_id": row["abnormality id"],
        "abnormality_type": row["abnormality type"],
        "pathology": row["pathology"],
        "full_mammogram_path": row["full_mammogram_path"],
        "roi_mask_path": row["roi_mask_path"],
    }

    try:
        full_path = resolve_path(row["full_mammogram_path"])
        roi_path = resolve_path(row["roi_mask_path"])

        if full_path is None or not full_path.exists():
            raise FileNotFoundError(f"Full mammogram not found: {full_path}")
        if roi_path is None or not roi_path.exists():
            raise FileNotFoundError(f"ROI mask not found: {roi_path}")

        image_array = load_dicom_pixels(full_path)
        roi_array = load_dicom_pixels(roi_path)
        mask_array = binarise_roi_mask(roi_array)
        mask_array, mask_resized = align_mask_to_full_image(mask_array, image_array.shape)

        roi_pixels = int(np.count_nonzero(mask_array))
        if roi_pixels < 2:
            raise ValueError("ROI contains fewer than two foreground pixels.")

        image_sitk, mask_sitk = numpy_to_sitk_image(image_array, mask_array)

        result = extractor.execute(
            image_sitk,
            mask_sitk,
            label=1,
        )
        features = extract_feature_columns(result)

        record = {
            **metadata,
            "extraction_status": "success",
            "roi_source_height": int(roi_array.shape[0]),
            "roi_source_width": int(roi_array.shape[1]),
            "full_image_height": int(image_array.shape[0]),
            "full_image_width": int(image_array.shape[1]),
            "roi_resized_to_full_image": bool(mask_resized),
            "roi_pixel_count": roi_pixels,
            "roi_fraction_of_full_image": float(roi_pixels / image_array.size),
            **features,
        }
        records.append(record)

    except Exception as exc:
        errors.append({
            **metadata,
            "extraction_status": "error",
            "error_message": str(exc),
        })

    completed = len(records) + len(errors)
    if completed % CHECKPOINT_EVERY == 0:
        write_checkpoint(records)
        if errors:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(errors).to_csv(ERROR_CSV, index=False)

write_checkpoint(records)
if errors:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(errors).to_csv(ERROR_CSV, index=False)

elapsed_seconds = time.time() - start_time
features_df = pd.DataFrame(records)
feature_columns = [c for c in features_df.columns if c.startswith("original_")]

feature_families = {}
for column in feature_columns:
    parts = column.split("_", 2)
    if len(parts) >= 2:
        family = parts[1]
        feature_families[family] = feature_families.get(family, 0) + 1

summary = {
    "total_requested": int(len(usable)),
    "successful_extractions": int(len(records)),
    "failed_extractions": int(len(errors)),
    "radiomics_feature_count": int(len(feature_columns)),
    "feature_count_by_family": feature_families,
    "elapsed_seconds": float(elapsed_seconds),
    "parameters": {
        "image_type": "Original",
        "force2D": True,
        "normalize": True,
        "normalizeScale": 100,
        "binWidth": 2,
        "resampledPixelSpacing": None,
    },
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print("\n" + "=" * 72)
print("PYRADIOMICS EXTRACTION SUMMARY")
print("=" * 72)
print(f"\nRequested lesion ROIs: {len(usable):,}")
print(f"Successful extractions: {len(records):,}")
print(f"Failed extractions: {len(errors):,}")
print(f"Radiomics features per lesion: {len(feature_columns):,}")
print("\nFeatures by family:")
for family, count in sorted(feature_families.items()):
    print(f"{family:12s}: {count}")

if len(features_df) > 0:
    print("\nROI masks resized to full-image dimensions:")
    print(features_df["roi_resized_to_full_image"].value_counts())

print(f"\nElapsed time: {elapsed_seconds / 60.0:.1f} minutes")
print("\nSaved lesion-level radiomics features:")
print(OUTPUT_CSV.resolve())
if errors:
    print("\nSaved extraction errors:")
    print(ERROR_CSV.resolve())
print("\nSaved extraction summary:")
print(SUMMARY_JSON.resolve())
