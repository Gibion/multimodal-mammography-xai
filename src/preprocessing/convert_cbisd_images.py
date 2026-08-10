from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

MANIFEST_FILE = "cbis_ddsm_resolved_manifest_v2.csv"

OUTPUT_ROOT = Path("../data/processed/cbis_ddsm")

FULL_DIR = OUTPUT_ROOT / "full_mammograms"
CROP_DIR = OUTPUT_ROOT / "cropped"
MASK_DIR = OUTPUT_ROOT / "roi_masks"

OUTPUT_MANIFEST = OUTPUT_ROOT / "cbis_ddsm_processed_manifest.csv"

JPEG_QUALITY = 95


# ============================================================
# Create folders
# ============================================================

FULL_DIR.mkdir(parents=True, exist_ok=True)
CROP_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Image utilities
# ============================================================

def normalize_to_uint8(image):
    """
    Convert an arbitrary grayscale pixel array to uint8
    using min-max normalization.

    This handles both 8-bit and 16-bit CBIS-DDSM images.
    """

    image = image.astype(np.float32)

    finite_mask = np.isfinite(image)

    if not finite_mask.any():
        raise ValueError("Image contains no finite pixel values.")

    image_min = image[finite_mask].min()
    image_max = image[finite_mask].max()

    if image_max <= image_min:
        return np.zeros(image.shape, dtype=np.uint8)

    image = (image - image_min) / (image_max - image_min)

    image = np.clip(image, 0.0, 1.0)

    image = image * 255.0

    return image.astype(np.uint8)


def read_dicom_pixels(dicom_path):
    """
    Read DICOM pixels and return the dataset plus pixel array.
    """

    ds = pydicom.dcmread(dicom_path)

    image = ds.pixel_array

    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale image, got shape {image.shape}"
        )

    return ds, image


def dicom_to_jpeg(dicom_path, output_path):
    """
    Convert mammogram/cropped-image DICOM to an 8-bit JPEG.
    """

    ds, image = read_dicom_pixels(dicom_path)

    image = normalize_to_uint8(image)

    photometric = str(
        getattr(ds, "PhotometricInterpretation", "")
    ).upper()

    # MONOCHROME1 means low values are displayed as white.
    # Convert it into the conventional MONOCHROME2 orientation.
    if photometric == "MONOCHROME1":
        image = 255 - image

    Image.fromarray(
        image,
        mode="L"
    ).save(
        output_path,
        format="JPEG",
        quality=JPEG_QUALITY,
        subsampling=0
    )


def dicom_mask_to_png(dicom_path, output_path):
    """
    Convert ROI-mask DICOM to a binary PNG.

    Background = 0
    ROI        = 255
    """

    ds, image = read_dicom_pixels(dicom_path)

    # Masks may not literally be encoded as 0/1.
    # Determine background from the most common pixel value.
    values, counts = np.unique(
        image,
        return_counts=True
    )

    background_value = values[
        np.argmax(counts)
    ]

    mask = image != background_value

    mask = (
        mask.astype(np.uint8) * 255
    )

    Image.fromarray(
        mask,
        mode="L"
    ).save(
        output_path,
        format="PNG"
    )


# ============================================================
# Filename generation
# ============================================================

def clean_component(value):
    """
    Make a value safe for filenames.
    """

    value = str(value).strip()

    value = value.replace(" ", "_")
    value = value.replace("/", "-")
    value = value.replace("\\", "-")

    return value


def create_case_stem(row):
    """
    Create a stable unique filename stem.

    Example:
    P_00005_RIGHT_CC_calcification_1
    """

    return "_".join([
        clean_component(row["patient_id"]),
        clean_component(row["left or right breast"]),
        clean_component(row["image view"]),
        clean_component(row["abnormality type"]),
        str(int(row["abnormality id"]))
    ])


# ============================================================
# Load manifest
# ============================================================

df = pd.read_csv(MANIFEST_FILE)

print(f"Manifest rows: {len(df):,}")


# ============================================================
# Prepare output columns
# ============================================================

df["full_jpeg_path"] = None
df["cropped_jpeg_path"] = None
df["roi_png_path"] = None

df["full_conversion_status"] = None
df["cropped_conversion_status"] = None
df["roi_conversion_status"] = None


# ============================================================
# Conversion
# ============================================================

for index, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Converting CBIS-DDSM"
):

    stem = create_case_stem(row)

    # --------------------------------------------------------
    # Full mammogram
    # --------------------------------------------------------

    if pd.notna(row["full_mammogram_path"]):

        output_path = FULL_DIR / f"{stem}_full.jpg"

        try:

            dicom_to_jpeg(
                row["full_mammogram_path"],
                output_path
            )

            df.at[index, "full_jpeg_path"] = str(
                output_path.resolve()
            )

            df.at[index, "full_conversion_status"] = "success"

        except Exception as exc:

            df.at[
                index,
                "full_conversion_status"
            ] = f"error: {exc}"

    else:

        df.at[
            index,
            "full_conversion_status"
        ] = "missing_source"


    # --------------------------------------------------------
    # Cropped image
    # --------------------------------------------------------

    if pd.notna(row["cropped_path"]):

        output_path = CROP_DIR / f"{stem}_cropped.jpg"

        try:

            dicom_to_jpeg(
                row["cropped_path"],
                output_path
            )

            df.at[index, "cropped_jpeg_path"] = str(
                output_path.resolve()
            )

            df.at[index, "cropped_conversion_status"] = "success"

        except Exception as exc:

            df.at[
                index,
                "cropped_conversion_status"
            ] = f"error: {exc}"

    else:

        df.at[
            index,
            "cropped_conversion_status"
        ] = "missing_source"


    # --------------------------------------------------------
    # ROI mask
    # --------------------------------------------------------

    if pd.notna(row["roi_mask_path"]):

        output_path = MASK_DIR / f"{stem}_roi.png"

        try:

            dicom_mask_to_png(
                row["roi_mask_path"],
                output_path
            )

            df.at[index, "roi_png_path"] = str(
                output_path.resolve()
            )

            df.at[index, "roi_conversion_status"] = "success"

        except Exception as exc:

            df.at[
                index,
                "roi_conversion_status"
            ] = f"error: {exc}"

    else:

        df.at[
            index,
            "roi_conversion_status"
        ] = "missing_source"


# ============================================================
# Save processed manifest
# ============================================================

df.to_csv(
    OUTPUT_MANIFEST,
    index=False
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 72)
print("CONVERSION SUMMARY")
print("=" * 72)

print("\nFull mammograms:")
print(
    df["full_conversion_status"]
    .value_counts(dropna=False)
)

print("\nCropped images:")
print(
    df["cropped_conversion_status"]
    .value_counts(dropna=False)
)

print("\nROI masks:")
print(
    df["roi_conversion_status"]
    .value_counts(dropna=False)
)

print("\nOutput folders:")
print(f"Full mammograms: {FULL_DIR.resolve()}")
print(f"Cropped images:  {CROP_DIR.resolve()}")
print(f"ROI masks:       {MASK_DIR.resolve()}")

print("\nProcessed manifest:")
print(OUTPUT_MANIFEST.resolve())