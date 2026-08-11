from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

MANIFEST_FILE = Path(
    "data/processed/cbis_ddsm/manifests/"
    "cbis_ddsm_full_image_manifest_split.csv"
)

OUTPUT_ROOT = Path(
    "data/processed/cbis_ddsm/model_input/full_mammograms_v2"
)

OUTPUT_MANIFEST = Path(
    "data/processed/cbis_ddsm/manifests/"
    "cbis_ddsm_full_image_manifest_preprocessed_v2.csv"
)

TARGET_SIZE = 512
CROP_MARGIN = 0.08
JPEG_QUALITY = 95
TARGET_CHEST_WALL_SIDE = "left"
CALIBRATION_SAMPLES_PER_LATERALITY = 150
RANDOM_STATE = 42
PROJECT_ROOT = Path.cwd()


def resolve_manifest_path(value):
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_grayscale(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def create_breast_mask(image):
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    _, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel = np.ones((7, 7), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if n_labels <= 1:
        raise ValueError("No breast foreground component detected.")

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(component_areas) + 1
    return (labels == largest_label).astype(np.uint8)


def detect_chest_wall_side(mask):
    _, width = mask.shape
    edge_width = max(1, int(width * 0.20))

    left_pixels = np.count_nonzero(mask[:, :edge_width])
    right_pixels = np.count_nonzero(mask[:, -edge_width:])

    return "left" if left_pixels >= right_pixels else "right"


def crop_to_breast(image, mask, margin_fraction=0.08):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Breast mask contains no foreground.")

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    image_height, image_width = image.shape
    box_width = x_max - x_min + 1
    box_height = y_max - y_min + 1

    margin_x = int(round(box_width * margin_fraction))
    margin_y = int(round(box_height * margin_fraction))

    x_min = max(0, x_min - margin_x)
    x_max = min(image_width - 1, x_max + margin_x)
    y_min = max(0, y_min - margin_y)
    y_max = min(image_height - 1, y_max + margin_y)

    cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]
    cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]

    return cropped_image, cropped_mask, (x_min, y_min, x_max, y_max)


def resize_and_pad(image, target_size=512):
    original_height, original_width = image.shape

    scale = min(
        target_size / original_width,
        target_size / original_height,
    )

    new_width = max(1, int(round(original_width * scale)))
    new_height = max(1, int(round(original_height * scale)))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)

    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width,
    ] = resized

    return canvas, scale, x_offset, y_offset, new_width, new_height


def create_image_id(path_value):
    return Path(str(path_value)).stem


def build_output_filename(row):
    patient = str(row["patient_id"]).strip()
    breast = str(row["left_or_right_breast"]).strip()
    view = str(row["image_view"]).strip()
    image_id = str(row["image_id"])[:8]
    return f"{patient}_{breast}_{view}_{image_id}.jpg"


def normalise_laterality(value):
    value = str(value).strip().upper()

    if value in {"L", "LEFT"}:
        return "LEFT"
    if value in {"R", "RIGHT"}:
        return "RIGHT"

    raise ValueError(f"Unexpected breast laterality value: {value}")


def calibrate_laterality_orientation(df):
    rng = np.random.default_rng(RANDOM_STATE)
    observations = {"LEFT": [], "RIGHT": []}

    print("\nCalibrating laterality orientation...")

    for laterality in ["LEFT", "RIGHT"]:
        subset = df[
            df["left_or_right_breast"]
            .astype(str)
            .str.upper()
            .eq(laterality)
        ]

        if len(subset) == 0:
            raise ValueError(f"No {laterality} images found for calibration.")

        sample_size = min(CALIBRATION_SAMPLES_PER_LATERALITY, len(subset))

        sampled_indices = rng.choice(
            subset.index.to_numpy(),
            size=sample_size,
            replace=False,
        )

        for idx in tqdm(
            sampled_indices,
            desc=f"Calibrating {laterality}",
            leave=False,
        ):
            row = df.loc[idx]
            source_path = resolve_manifest_path(row["full_jpeg_path"])
            image = load_grayscale(source_path)
            mask = create_breast_mask(image)
            observations[laterality].append(detect_chest_wall_side(mask))

    mapping = {}

    print("\nCalibration results:")

    for laterality, sides in observations.items():
        counts = Counter(sides)
        dominant_side = counts.most_common(1)[0][0]
        mapping[laterality] = dominant_side

        print(
            f"{laterality:5s}: "
            f"left={counts.get('left', 0):3d}, "
            f"right={counts.get('right', 0):3d}, "
            f"dominant={dominant_side}"
        )

    if mapping["LEFT"] == mapping["RIGHT"]:
        print(
            "\nWARNING: LEFT and RIGHT mammograms have the same dominant "
            "detected orientation. Review the calibration results before "
            "using the metadata-based flip rule."
        )

    return mapping


def apply_metadata_orientation(image, mask, laterality, laterality_mapping):
    laterality = normalise_laterality(laterality)
    expected_raw_side = laterality_mapping[laterality]

    flipped = expected_raw_side != TARGET_CHEST_WALL_SIDE

    if flipped:
        image = np.fliplr(image)
        mask = np.fliplr(mask)

    return image, mask, expected_raw_side, flipped


# ============================================================
# Load manifest
# ============================================================

df = pd.read_csv(MANIFEST_FILE)

print("=" * 72)
print("FULL-MAMMOGRAM PREPROCESSING V2")
print("=" * 72)

print(f"\nManifest rows: {len(df):,}")

print("\nSplits:")
print(df["split"].value_counts())


# ============================================================
# Stable unique image IDs
# ============================================================

df["image_id"] = df["full_mammogram_path"].apply(create_image_id)

print("\nUnique image IDs:", df["image_id"].nunique(), "/", len(df))

if df["image_id"].nunique() != len(df):
    raise ValueError("image_id is not unique.")


# ============================================================
# Verify source paths
# ============================================================

source_paths = df["full_jpeg_path"].apply(resolve_manifest_path)
existing = source_paths.apply(lambda p: p.exists())

print("\nSource image verification:")
print(f"Found: {existing.sum():,} / {len(existing):,}")

if not existing.all():
    missing = source_paths[~existing]

    print("\nFirst missing paths:")
    for path in missing.head(10):
        print(path)

    raise FileNotFoundError(
        f"{(~existing).sum():,} source mammograms could not be found."
    )


# ============================================================
# Unique output filenames
# ============================================================

df["output_filename"] = df.apply(build_output_filename, axis=1)

duplicate_names = df["output_filename"].duplicated(keep=False)

if duplicate_names.any():
    print("\nDuplicate output filenames:")
    print(
        df.loc[
            duplicate_names,
            [
                "patient_id",
                "left_or_right_breast",
                "image_view",
                "full_mammogram_path",
                "output_filename",
            ],
        ].to_string(index=False)
    )
    raise ValueError("Output filenames are not unique.")

print("\nUnique output filenames:", df["output_filename"].nunique(), "/", len(df))


# ============================================================
# Calibrate metadata-driven orientation rule
# ============================================================

laterality_mapping = calibrate_laterality_orientation(df)

print("\nFinal metadata orientation rule:")

for laterality in ["LEFT", "RIGHT"]:
    raw_side = laterality_mapping[laterality]
    action = "flip" if raw_side != TARGET_CHEST_WALL_SIDE else "do not flip"

    print(
        f"{laterality:5s}: "
        f"dominant raw chest wall={raw_side}; "
        f"{action}"
    )


# ============================================================
# Prepare output columns
# ============================================================

output_columns = [
    "preprocessed_path",
    "preprocessing_status",
    "original_width",
    "original_height",
    "cropped_width",
    "cropped_height",
    "crop_x_min",
    "crop_y_min",
    "crop_x_max",
    "crop_y_max",
    "laterality",
    "expected_raw_chest_wall",
    "orientation_flipped",
    "qa_detected_chest_wall_before",
    "qa_detected_chest_wall_after",
    "resize_scale",
    "resized_width",
    "resized_height",
    "padding_x",
    "padding_y",
    "foreground_fraction_before_crop",
    "foreground_fraction_after_crop",
]

for column in output_columns:
    if column not in df.columns:
        df[column] = None


# ============================================================
# Preprocess
# ============================================================

for index, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Preprocessing mammograms",
):
    try:
        source_path = resolve_manifest_path(row["full_jpeg_path"])
        image = load_grayscale(source_path)
        original_height, original_width = image.shape

        mask = create_breast_mask(image)

        foreground_fraction_before = np.count_nonzero(mask) / mask.size
        qa_side_before = detect_chest_wall_side(mask)

        image, mask, crop_box = crop_to_breast(
            image,
            mask,
            CROP_MARGIN,
        )

        cropped_height, cropped_width = image.shape
        foreground_fraction_after = np.count_nonzero(mask) / mask.size

        laterality = normalise_laterality(row["left_or_right_breast"])

        image, mask, expected_raw_side, flipped = apply_metadata_orientation(
            image,
            mask,
            laterality,
            laterality_mapping,
        )

        qa_side_after = detect_chest_wall_side(mask)

        (
            image,
            scale,
            padding_x,
            padding_y,
            resized_width,
            resized_height,
        ) = resize_and_pad(image, TARGET_SIZE)

        split = str(row["split"])
        output_directory = OUTPUT_ROOT / split
        output_directory.mkdir(parents=True, exist_ok=True)

        output_path = output_directory / row["output_filename"]

        Image.fromarray(image, mode="L").save(
            output_path,
            format="JPEG",
            quality=JPEG_QUALITY,
            subsampling=0,
        )

        x_min, y_min, x_max, y_max = crop_box

        df.at[index, "preprocessed_path"] = str(output_path)
        df.at[index, "preprocessing_status"] = "success"

        df.at[index, "original_width"] = original_width
        df.at[index, "original_height"] = original_height
        df.at[index, "cropped_width"] = cropped_width
        df.at[index, "cropped_height"] = cropped_height

        df.at[index, "crop_x_min"] = x_min
        df.at[index, "crop_y_min"] = y_min
        df.at[index, "crop_x_max"] = x_max
        df.at[index, "crop_y_max"] = y_max

        df.at[index, "laterality"] = laterality
        df.at[index, "expected_raw_chest_wall"] = expected_raw_side
        df.at[index, "orientation_flipped"] = flipped
        df.at[index, "qa_detected_chest_wall_before"] = qa_side_before
        df.at[index, "qa_detected_chest_wall_after"] = qa_side_after

        df.at[index, "resize_scale"] = scale
        df.at[index, "resized_width"] = resized_width
        df.at[index, "resized_height"] = resized_height
        df.at[index, "padding_x"] = padding_x
        df.at[index, "padding_y"] = padding_y

        df.at[index, "foreground_fraction_before_crop"] = foreground_fraction_before
        df.at[index, "foreground_fraction_after_crop"] = foreground_fraction_after

    except Exception as exc:
        df.at[index, "preprocessing_status"] = f"error: {exc}"


# ============================================================
# Save
# ============================================================

OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_MANIFEST, index=False)


# ============================================================
# Summary
# ============================================================

print("\n")
print("=" * 72)
print("PREPROCESSING V2 SUMMARY")
print("=" * 72)

print("\nStatus:")
print(df["preprocessing_status"].value_counts(dropna=False))

successful = df[df["preprocessing_status"] == "success"].copy()

print("\nSuccessfully processed:")
print(f"{len(successful):,} / {len(df):,}")

print("\nLaterality:")
print(successful["laterality"].value_counts())

print("\nMetadata-driven flips:")
print(successful["orientation_flipped"].value_counts())

print("\nQA chest-wall detection BEFORE normalisation:")
print(successful["qa_detected_chest_wall_before"].value_counts())

print("\nQA chest-wall detection AFTER normalisation:")
print(successful["qa_detected_chest_wall_after"].value_counts())

print("\nQA disagreement after normalisation:")
qa_disagreement = (
    successful["qa_detected_chest_wall_after"]
    != TARGET_CHEST_WALL_SIDE
)
print(f"{qa_disagreement.sum():,} / {len(successful):,}")

print("\nForeground fraction BEFORE crop:")
print(
    pd.to_numeric(
        successful["foreground_fraction_before_crop"]
    ).describe()
)

print("\nForeground fraction AFTER crop:")
print(
    pd.to_numeric(
        successful["foreground_fraction_after_crop"]
    ).describe()
)

print("\nOutput images:")
print(successful["split"].value_counts())

print("\nSaved manifest:")
print(OUTPUT_MANIFEST.resolve())

print("\nOutput root:")
print(OUTPUT_ROOT.resolve())
