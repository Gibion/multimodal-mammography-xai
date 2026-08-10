from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


MANIFEST_FILE = "../data/processed/cbis_ddsm/cbis_ddsm_processed_manifest.csv"
OUTPUT_FILE = "../data/processed/cbis_ddsm/cbis_ddsm_qa_manifest.csv"


df = pd.read_csv(MANIFEST_FILE)

qa_records = []


def inspect_image(path):
    """
    Open an image and return basic QA information.
    """

    with Image.open(path) as img:
        array = np.asarray(img)

        return {
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "min_pixel": int(array.min()),
            "max_pixel": int(array.max()),
            "mean_pixel": float(array.mean()),
            "std_pixel": float(array.std()),
        }


for index, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Checking processed images"
):

    record = {
        "source_row": index,
        "dataset": row["dataset"],
        "patient_id": row["patient_id"],
        "abnormality_id": row["abnormality id"],
    }

    # ------------------------------------------------------
    # Full mammogram
    # ------------------------------------------------------

    full_path = row.get("full_jpeg_path")

    if pd.notna(full_path):

        try:
            stats = inspect_image(full_path)

            record["full_valid"] = True
            record["full_width"] = stats["width"]
            record["full_height"] = stats["height"]
            record["full_min"] = stats["min_pixel"]
            record["full_max"] = stats["max_pixel"]
            record["full_mean"] = stats["mean_pixel"]
            record["full_std"] = stats["std_pixel"]

        except Exception as exc:

            record["full_valid"] = False
            record["full_error"] = str(exc)

    else:
        record["full_valid"] = False


    # ------------------------------------------------------
    # Cropped image
    # ------------------------------------------------------

    crop_path = row.get("cropped_jpeg_path")

    if pd.notna(crop_path):

        try:
            stats = inspect_image(crop_path)

            record["cropped_valid"] = True
            record["cropped_width"] = stats["width"]
            record["cropped_height"] = stats["height"]
            record["cropped_min"] = stats["min_pixel"]
            record["cropped_max"] = stats["max_pixel"]
            record["cropped_mean"] = stats["mean_pixel"]
            record["cropped_std"] = stats["std_pixel"]

        except Exception as exc:

            record["cropped_valid"] = False
            record["cropped_error"] = str(exc)

    else:
        record["cropped_valid"] = False


    # ------------------------------------------------------
    # ROI mask
    # ------------------------------------------------------

    mask_path = row.get("roi_png_path")

    if pd.notna(mask_path):

        try:

            with Image.open(mask_path) as img:
                mask = np.asarray(img)

            unique_values = np.unique(mask)

            record["roi_valid"] = True
            record["roi_width"] = img.width
            record["roi_height"] = img.height
            record["roi_unique_values"] = ",".join(
                str(int(v)) for v in unique_values
            )

            record["roi_foreground_pixels"] = int(
                np.count_nonzero(mask)
            )

            record["roi_has_foreground"] = (
                np.count_nonzero(mask) > 0
            )

        except Exception as exc:

            record["roi_valid"] = False
            record["roi_error"] = str(exc)

    else:

        record["roi_valid"] = False
        record["roi_has_foreground"] = False


    qa_records.append(record)


qa = pd.DataFrame(qa_records)

qa.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nQA SUMMARY")
print("=" * 70)

print("\nFull mammograms valid:")
print(qa["full_valid"].value_counts(dropna=False))

print("\nCropped images valid:")
print(qa["cropped_valid"].value_counts(dropna=False))

print("\nROI masks valid:")
print(qa["roi_valid"].value_counts(dropna=False))

print("\nROI masks with foreground:")
print(
    qa.loc[
        qa["roi_valid"] == True,
        "roi_has_foreground"
    ].value_counts(dropna=False)
)

print("\nPotentially blank full mammograms:")
print(
    (
        (qa["full_std"] < 1)
        & qa["full_valid"]
    ).sum()
)

print("\nPotentially blank cropped images:")
print(
    (
        (qa["cropped_std"] < 1)
        & qa["cropped_valid"]
    ).sum()
)

print("\nSaved:")
print(OUTPUT_FILE)