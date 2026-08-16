from pathlib import Path
import hashlib
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

try:
    import pydicom
except ImportError:
    pydicom = None


PROJECT_ROOT = Path.cwd()

FULL_MANIFEST = Path(
    "data/processed/cbis_ddsm/manifests/"
    "cbis_ddsm_full_image_manifest_preprocessed_v4.csv"
)

LESION_MANIFEST_CANDIDATES = [
    Path(
        "data/processed/cbis_ddsm/manifests/"
        "cbis_ddsm_processed_manifest.csv"
    ),
    Path(
        "data/processed/cbis_ddsm/"
        "cbis_ddsm_processed_manifest.csv"
    ),
]

OUTPUT_ROOT = Path(
    "data/processed/cbis_ddsm/xai/roi_masks_512"
)

OUTPUT_MANIFEST = Path(
    "data/processed/cbis_ddsm/manifests/"
    "cbis_ddsm_roi_xai_manifest.csv"
)

QA_FIGURE_DIR = Path(
    "results/figures/xai_roi_preparation"
)

TARGET_SIZE = 512
QA_EXAMPLES = 20
RANDOM_SEED = 42


def find_existing_file(candidates):
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find the lesion-level processed manifest. Checked:\n"
        + "\n".join(str(path.resolve()) for path in candidates)
    )


def resolve_path(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def first_existing_column(frame, candidates):
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def normalise_key(value):
    if pd.isna(value):
        return None
    return str(resolve_path(value))


def to_uint8(array):
    array = np.asarray(array)
    if array.size == 0:
        raise ValueError("Image array is empty.")

    array = np.nan_to_num(
        array,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    minimum = float(np.min(array))
    maximum = float(np.max(array))

    if maximum <= minimum:
        return np.zeros(array.shape, dtype=np.uint8)

    scaled = (
        (array.astype(np.float32) - minimum)
        / (maximum - minimum)
        * 255.0
    )

    return np.clip(scaled, 0, 255).astype(np.uint8)


def load_roi_image(path):
    suffix = path.suffix.lower()

    if suffix == ".dcm":
        if pydicom is None:
            raise ImportError(
                "pydicom is required to read DICOM ROI masks."
            )
        ds = pydicom.dcmread(str(path))
        return np.asarray(ds.pixel_array)

    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def corner_background_is_white(binary_255):
    height, width = binary_255.shape
    patch_h = max(1, int(round(height * 0.05)))
    patch_w = max(1, int(round(width * 0.05)))

    patches = [
        binary_255[:patch_h, :patch_w],
        binary_255[:patch_h, width - patch_w:],
        binary_255[height - patch_h:, :patch_w],
        binary_255[height - patch_h:, width - patch_w:],
    ]

    corner_mean = np.mean(
        [float(patch.mean()) for patch in patches]
    )

    return corner_mean > 127.5


def binarise_roi_mask(array):
    image = to_uint8(array)

    if np.max(image) == 0:
        return np.zeros(image.shape, dtype=np.uint8), False

    _, thresholded = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    inverted = False

    if corner_background_is_white(thresholded):
        thresholded = 255 - thresholded
        inverted = True

    return (thresholded > 0).astype(np.uint8), inverted


def resize_binary_mask(mask, target_width, target_height):
    return cv2.resize(
        mask.astype(np.uint8),
        (int(target_width), int(target_height)),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.uint8)


def transform_mask_to_v4_space(mask, full_row):
    original_width = int(full_row["original_width"])
    original_height = int(full_row["original_height"])

    source_height, source_width = mask.shape

    dimensions_matched = (
        source_width == original_width
        and source_height == original_height
    )

    if not dimensions_matched:
        mask = resize_binary_mask(
            mask,
            original_width,
            original_height
        )

    x_min = int(full_row["crop_x_min"])
    y_min = int(full_row["crop_y_min"])
    x_max = int(full_row["crop_x_max"])
    y_max = int(full_row["crop_y_max"])

    cropped = mask[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    orientation_flipped = str(
        full_row["orientation_flipped"]
    ).strip().lower() in {"true", "1", "yes"}

    if orientation_flipped:
        cropped = np.fliplr(cropped).copy()

    resized_width = int(full_row["resized_width"])
    resized_height = int(full_row["resized_height"])

    resized = resize_binary_mask(
        cropped,
        resized_width,
        resized_height
    )

    padding_x = int(full_row["padding_x"])
    padding_y = int(full_row["padding_y"])

    canvas = np.zeros(
        (TARGET_SIZE, TARGET_SIZE),
        dtype=np.uint8
    )

    x_end = padding_x + resized_width
    y_end = padding_y + resized_height

    if x_end > TARGET_SIZE or y_end > TARGET_SIZE:
        raise ValueError(
            "Recorded V4 resize/padding values exceed the 512 x 512 canvas."
        )

    canvas[
        padding_y:y_end,
        padding_x:x_end
    ] = resized

    return (
        (canvas > 0).astype(np.uint8),
        dimensions_matched,
        source_width,
        source_height
    )


def make_roi_id(row):
    parts = [
        str(row.get("dataset", "")),
        str(row.get("source_row", "")),
        str(row.get("patient_id", "")),
        str(
            row.get(
                "abnormality id",
                row.get("abnormality_id", "")
            )
        ),
        str(row.get("full_image_id", "")),
    ]

    return hashlib.sha1(
        "|".join(parts).encode("utf-8")
    ).hexdigest()[:10]


def save_mask_png(mask, output_path):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Image.fromarray(
        (mask * 255).astype(np.uint8),
        mode="L"
    ).save(
        output_path,
        format="PNG"
    )


lesion_manifest_path = find_existing_file(
    LESION_MANIFEST_CANDIDATES
)

full_df = pd.read_csv(FULL_MANIFEST)
lesion_df = pd.read_csv(lesion_manifest_path)

print("=" * 72)
print("PREPARE CBIS-DDSM ROI MASKS FOR XAI")
print("=" * 72)

print(f"\nFull-image manifest rows: {len(full_df):,}")
print(f"Lesion-level manifest rows: {len(lesion_df):,}")
print("\nLesion manifest:")
print(lesion_manifest_path.resolve())


required_full_columns = [
    "image_id",
    "split",
    "preprocessed_path",
    "original_width",
    "original_height",
    "crop_x_min",
    "crop_y_min",
    "crop_x_max",
    "crop_y_max",
    "orientation_flipped",
    "resized_width",
    "resized_height",
    "padding_x",
    "padding_y",
]

missing_full_columns = [
    column
    for column in required_full_columns
    if column not in full_df.columns
]

if missing_full_columns:
    raise ValueError(
        "The V4 manifest is missing required transform columns: "
        + ", ".join(missing_full_columns)
    )


join_column = None

for candidate in [
    "full_mammogram_path",
    "full_jpeg_path",
]:
    if candidate in full_df.columns and candidate in lesion_df.columns:
        join_column = candidate
        break

if join_column is None:
    raise ValueError(
        "Could not find a common full-image path column. "
        "Expected full_mammogram_path or full_jpeg_path."
    )

print(f"\nJoining manifests using: {join_column}")

full_df["_full_join_key"] = full_df[
    join_column
].apply(normalise_key)

lesion_df["_full_join_key"] = lesion_df[
    join_column
].apply(normalise_key)

if full_df["_full_join_key"].duplicated(keep=False).any():
    raise ValueError(
        "The V4 manifest contains duplicate full-image join keys."
    )


roi_column_candidates = [
    "roi_mask_jpeg_path",
    "roi_jpeg_path",
    "processed_roi_mask_path",
    "roi_mask_processed_path",
    "roi_mask_path",
]

roi_source_column = first_existing_column(
    lesion_df,
    roi_column_candidates
)

if roi_source_column is None:
    raise ValueError(
        "Could not find an ROI-mask path column. Checked: "
        + ", ".join(roi_column_candidates)
    )

print(f"ROI source column: {roi_source_column}")


full_columns_to_merge = [
    "_full_join_key",
    "image_id",
    "split",
    "preprocessed_path",
    "original_width",
    "original_height",
    "crop_x_min",
    "crop_y_min",
    "crop_x_max",
    "crop_y_max",
    "orientation_flipped",
    "resized_width",
    "resized_height",
    "padding_x",
    "padding_y",
]

full_lookup = full_df[
    full_columns_to_merge
].copy().rename(
    columns={
        "image_id": "full_image_id",
        "split": "full_split",
        "preprocessed_path": "full_preprocessed_path",
    }
)

merged = lesion_df.merge(
    full_lookup,
    on="_full_join_key",
    how="left",
    validate="many_to_one",
    indicator=True
)

print("\nFull-image matching:")
print(merged["_merge"].value_counts())


records = []

for _, row in merged.iterrows():
    record = row.to_dict()
    roi_value = row.get(roi_source_column)

    if pd.isna(roi_value) or not str(roi_value).strip():
        record.update({
            "roi_xai_status": "missing_roi_source",
            "roi_source_resolved": None,
            "roi_mask_512_path": None,
            "roi_foreground_pixels_source": 0,
            "roi_foreground_pixels_512": 0,
        })
        records.append(record)
        continue

    if row["_merge"] != "both":
        record.update({
            "roi_xai_status": "full_image_not_matched",
            "roi_source_resolved": None,
            "roi_mask_512_path": None,
            "roi_foreground_pixels_source": 0,
            "roi_foreground_pixels_512": 0,
        })
        records.append(record)
        continue

    try:
        roi_path = resolve_path(roi_value)

        if roi_path is None or not roi_path.exists():
            raise FileNotFoundError(
                f"ROI source not found: {roi_path}"
            )

        roi_array = load_roi_image(roi_path)

        roi_mask, polarity_inverted = binarise_roi_mask(
            roi_array
        )

        source_foreground = int(
            np.count_nonzero(roi_mask)
        )

        if source_foreground == 0:
            raise ValueError(
                "ROI mask has no foreground after binarisation."
            )

        (
            transformed_mask,
            dimensions_matched,
            roi_source_width,
            roi_source_height
        ) = transform_mask_to_v4_space(
            roi_mask,
            row
        )

        transformed_foreground = int(
            np.count_nonzero(
                transformed_mask
            )
        )

        if transformed_foreground == 0:
            raise ValueError(
                "ROI foreground disappeared after V4 transformation."
            )

        roi_id = make_roi_id(row)

        abnormality_id = row.get(
            "abnormality id",
            row.get("abnormality_id", "NA")
        )

        lesion_type = row.get(
            "abnormality type",
            row.get("abnormality_type", "lesion")
        )

        safe_lesion_type = (
            str(lesion_type)
            .strip()
            .replace(" ", "_")
            .replace("/", "_")
        )

        split = str(row["full_split"])

        output_filename = (
            f"{row['full_image_id']}_"
            f"abn{abnormality_id}_"
            f"{safe_lesion_type}_"
            f"{roi_id}.png"
        )

        output_path = (
            OUTPUT_ROOT
            / split
            / output_filename
        )

        save_mask_png(
            transformed_mask,
            output_path
        )

        ys, xs = np.where(
            transformed_mask > 0
        )

        record.update({
            "roi_xai_status": "success",
            "roi_source_resolved": str(roi_path),
            "roi_source_width": int(roi_source_width),
            "roi_source_height": int(roi_source_height),
            "roi_source_matches_full_dimensions": bool(
                dimensions_matched
            ),
            "roi_polarity_inverted": bool(
                polarity_inverted
            ),
            "roi_foreground_pixels_source": source_foreground,
            "roi_foreground_pixels_512": transformed_foreground,
            "roi_foreground_fraction_512": float(
                transformed_foreground
                / (TARGET_SIZE * TARGET_SIZE)
            ),
            "roi_bbox_x_min_512": int(xs.min()),
            "roi_bbox_y_min_512": int(ys.min()),
            "roi_bbox_x_max_512": int(xs.max()),
            "roi_bbox_y_max_512": int(ys.max()),
            "roi_centroid_x_512": float(xs.mean()),
            "roi_centroid_y_512": float(ys.mean()),
            "roi_mask_512_path": str(
                output_path.resolve()
            ),
        })

    except Exception as exc:
        record.update({
            "roi_xai_status": f"error: {exc}",
            "roi_source_resolved": str(
                resolve_path(roi_value)
            ) if not pd.isna(roi_value) else None,
            "roi_mask_512_path": None,
            "roi_foreground_pixels_512": 0,
        })

    records.append(record)


result_df = pd.DataFrame(records)

OUTPUT_MANIFEST.parent.mkdir(
    parents=True,
    exist_ok=True
)

result_df.to_csv(
    OUTPUT_MANIFEST,
    index=False
)


print("\n")
print("=" * 72)
print("ROI XAI PREPARATION SUMMARY")
print("=" * 72)

print("\nStatus:")
print(
    result_df[
        "roi_xai_status"
    ].value_counts(
        dropna=False
    )
)

successful = result_df[
    result_df["roi_xai_status"] == "success"
].copy()

print(
    f"\nSuccessfully prepared ROI masks: "
    f"{len(successful):,}"
)

if len(successful) > 0:
    print(
        "\nROI source dimensions already matched "
        "the full mammogram:"
    )
    print(
        successful[
            "roi_source_matches_full_dimensions"
        ].value_counts()
    )

    print("\nROI masks whose polarity was inverted:")
    print(
        successful[
            "roi_polarity_inverted"
        ].value_counts()
    )

    print(
        "\nROI foreground fraction in 512 x 512 space:"
    )
    print(
        pd.to_numeric(
            successful[
                "roi_foreground_fraction_512"
            ]
        ).describe()
    )

    print("\nSuccessful ROI masks by split:")
    print(
        successful[
            "full_split"
        ].value_counts()
    )


QA_FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

if len(successful) > 0:
    qa_count = min(
        QA_EXAMPLES,
        len(successful)
    )

    qa_sample = successful.sample(
        n=qa_count,
        random_state=RANDOM_SEED
    )

    columns = 4
    rows = math.ceil(
        qa_count / columns
    )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4 * columns, 4 * rows)
    )

    axes = np.atleast_1d(
        axes
    ).flatten()

    for ax in axes:
        ax.axis("off")

    for ax, (_, row) in zip(
        axes,
        qa_sample.iterrows()
    ):
        mammogram_path = resolve_path(
            row["full_preprocessed_path"]
        )

        roi_mask_path = Path(
            row["roi_mask_512_path"]
        )

        mammogram = np.asarray(
            Image.open(
                mammogram_path
            ).convert("L")
        )

        roi_mask = (
            np.asarray(
                Image.open(
                    roi_mask_path
                ).convert("L")
            ) > 0
        )

        ax.imshow(
            mammogram,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        masked_overlay = np.ma.masked_where(
            ~roi_mask,
            roi_mask
        )

        ax.imshow(
            masked_overlay,
            alpha=0.50,
            vmin=0,
            vmax=1
        )

        ax.set_title(
            f"{row.get('patient_id', '')} | "
            f"{row.get('abnormality type', row.get('abnormality_type', ''))}\n"
            f"{row.get('pathology', '')}",
            fontsize=9
        )

    fig.suptitle(
        "CBIS-DDSM ROI Masks Aligned to V4 Model-Input Space",
        fontsize=15
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.97]
    )

    qa_figure_path = (
        QA_FIGURE_DIR
        / "roi_alignment_examples.png"
    )

    fig.savefig(
        qa_figure_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "\nSaved visual alignment QA figure:"
    )
    print(
        qa_figure_path.resolve()
    )


print("\nSaved ROI XAI manifest:")
print(OUTPUT_MANIFEST.resolve())

print("\nSaved transformed ROI masks:")
print(OUTPUT_ROOT.resolve())
