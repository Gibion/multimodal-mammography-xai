from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

MANIFEST_FILE = Path(
    "data/processed/cbis_ddsm/manifests/"
    "cbis_ddsm_full_image_manifest_split.csv"
)

OUTPUT_ROOT = Path(
    "data/processed/cbis_ddsm/model_input/full_mammograms_v3"
)

OUTPUT_MANIFEST = Path(
    "data/processed/cbis_ddsm/manifests/"
    "cbis_ddsm_full_image_manifest_preprocessed_v3.csv"
)

TARGET_SIZE = 512
JPEG_QUALITY = 95

SAFETY_MARGIN_FRACTION = 0.18
EDGE_PROTECTION_FRACTION = 0.05
MAX_CROP_FRACTION_PER_SIDE = 0.25

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
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((7, 7), dtype=np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
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


def conservative_crop(image, mask, laterality):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Breast mask contains no foreground.")

    height, width = image.shape

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    breast_width = x_max - x_min + 1
    breast_height = y_max - y_min + 1

    margin_x = int(round(breast_width * SAFETY_MARGIN_FRACTION))
    margin_y = int(round(breast_height * SAFETY_MARGIN_FRACTION))

    crop_x_min = max(0, x_min - margin_x)
    crop_x_max = min(width - 1, x_max + margin_x)
    crop_y_min = max(0, y_min - margin_y)
    crop_y_max = min(height - 1, y_max + margin_y)

    edge_x = int(round(width * EDGE_PROTECTION_FRACTION))
    edge_y = int(round(height * EDGE_PROTECTION_FRACTION))

    if x_min <= edge_x:
        crop_x_min = 0
    if (width - 1 - x_max) <= edge_x:
        crop_x_max = width - 1
    if y_min <= edge_y:
        crop_y_min = 0
    if (height - 1 - y_max) <= edge_y:
        crop_y_max = height - 1

    laterality = str(laterality).strip().upper()

    # Protect the raw chest-wall edge completely.
    if laterality == "LEFT":
        crop_x_min = 0
    elif laterality == "RIGHT":
        crop_x_max = width - 1
    else:
        raise ValueError(f"Unexpected laterality: {laterality}")

    max_crop_x = int(round(width * MAX_CROP_FRACTION_PER_SIDE))
    max_crop_y = int(round(height * MAX_CROP_FRACTION_PER_SIDE))

    crop_x_min = min(crop_x_min, max_crop_x)
    crop_x_max = max(crop_x_max, width - 1 - max_crop_x)
    crop_y_min = min(crop_y_min, max_crop_y)
    crop_y_max = max(crop_y_max, height - 1 - max_crop_y)

    cropped_image = image[
        crop_y_min:crop_y_max + 1,
        crop_x_min:crop_x_max + 1
    ]

    cropped_mask = mask[
        crop_y_min:crop_y_max + 1,
        crop_x_min:crop_x_max + 1
    ]

    crop_fractions = {
        "crop_left_fraction": crop_x_min / width,
        "crop_right_fraction": (width - 1 - crop_x_max) / width,
        "crop_top_fraction": crop_y_min / height,
        "crop_bottom_fraction": (height - 1 - crop_y_max) / height,
    }

    return (
        cropped_image,
        cropped_mask,
        (crop_x_min, crop_y_min, crop_x_max, crop_y_max),
        crop_fractions
    )


def resize_and_pad(image, target_size=512):
    original_height, original_width = image.shape

    scale = min(
        target_size / original_width,
        target_size / original_height
    )

    new_width = max(1, int(round(original_width * scale)))
    new_height = max(1, int(round(original_height * scale)))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros(
        (target_size, target_size),
        dtype=np.uint8
    )

    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    return (
        canvas,
        scale,
        x_offset,
        y_offset,
        new_width,
        new_height
    )


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

    raise ValueError(f"Unexpected laterality value: {value}")


def calibrate_laterality_orientation(df):
    rng = np.random.default_rng(RANDOM_STATE)

    observations = {
        "LEFT": [],
        "RIGHT": []
    }

    print("\nCalibrating laterality orientation...")

    for laterality in ["LEFT", "RIGHT"]:

        subset = df[
            df["left_or_right_breast"]
            .astype(str)
            .str.upper()
            .eq(laterality)
        ]

        sample_size = min(
            CALIBRATION_SAMPLES_PER_LATERALITY,
            len(subset)
        )

        sampled_indices = rng.choice(
            subset.index.to_numpy(),
            size=sample_size,
            replace=False
        )

        for idx in tqdm(
            sampled_indices,
            desc=f"Calibrating {laterality}",
            leave=False
        ):
            row = df.loc[idx]
            source_path = resolve_manifest_path(
                row["full_jpeg_path"]
            )

            image = load_grayscale(source_path)
            mask = create_breast_mask(image)

            observations[laterality].append(
                detect_chest_wall_side(mask)
            )

    mapping = {}

    print("\nCalibration results:")

    for laterality, sides in observations.items():
        counts = Counter(sides)
        dominant = counts.most_common(1)[0][0]
        mapping[laterality] = dominant

        print(
            f"{laterality:5s}: "
            f"left={counts.get('left', 0):3d}, "
            f"right={counts.get('right', 0):3d}, "
            f"dominant={dominant}"
        )

    return mapping


def apply_metadata_orientation(
    image,
    mask,
    laterality,
    laterality_mapping
):
    laterality = normalise_laterality(laterality)

    expected_raw_side = laterality_mapping[laterality]

    flipped = (
        expected_raw_side != TARGET_CHEST_WALL_SIDE
    )

    if flipped:
        image = np.fliplr(image)
        mask = np.fliplr(mask)

    return (
        image,
        mask,
        expected_raw_side,
        flipped
    )


df = pd.read_csv(MANIFEST_FILE)

print("=" * 72)
print("FULL-MAMMOGRAM PREPROCESSING V3")
print("=" * 72)

print(f"\nManifest rows: {len(df):,}")

print("\nSplits:")
print(df["split"].value_counts())

df["image_id"] = df[
    "full_mammogram_path"
].apply(create_image_id)

print(
    "\nUnique image IDs:",
    df["image_id"].nunique(),
    "/",
    len(df)
)

if df["image_id"].nunique() != len(df):
    raise ValueError("image_id is not unique.")

source_paths = df[
    "full_jpeg_path"
].apply(resolve_manifest_path)

existing = source_paths.apply(
    lambda p: p.exists()
)

print("\nSource image verification:")
print(
    f"Found: {existing.sum():,} / {len(existing):,}"
)

if not existing.all():
    raise FileNotFoundError(
        f"{(~existing).sum():,} source mammograms missing."
    )

df["output_filename"] = df.apply(
    build_output_filename,
    axis=1
)

if df["output_filename"].duplicated().any():
    raise ValueError(
        "Output filenames are not unique."
    )

print(
    "\nUnique output filenames:",
    df["output_filename"].nunique(),
    "/",
    len(df)
)

laterality_mapping = calibrate_laterality_orientation(df)

print("\nFinal metadata orientation rule:")

for laterality in ["LEFT", "RIGHT"]:
    raw_side = laterality_mapping[laterality]
    action = (
        "flip"
        if raw_side != TARGET_CHEST_WALL_SIDE
        else "do not flip"
    )

    print(
        f"{laterality:5s}: "
        f"dominant raw chest wall={raw_side}; "
        f"{action}"
    )

output_columns = [
    "preprocessed_path",
    "preprocessing_status",
    "laterality",
    "orientation_flipped",
    "expected_raw_chest_wall",
    "qa_detected_chest_wall_before",
    "qa_detected_chest_wall_after",
    "original_width",
    "original_height",
    "cropped_width",
    "cropped_height",
    "crop_x_min",
    "crop_y_min",
    "crop_x_max",
    "crop_y_max",
    "crop_left_fraction",
    "crop_right_fraction",
    "crop_top_fraction",
    "crop_bottom_fraction",
    "total_removed_fraction",
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

for index, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Preprocessing mammograms"
):

    try:

        source_path = resolve_manifest_path(
            row["full_jpeg_path"]
        )

        image = load_grayscale(source_path)

        original_height, original_width = image.shape

        laterality = normalise_laterality(
            row["left_or_right_breast"]
        )

        mask = create_breast_mask(image)

        foreground_before = (
            np.count_nonzero(mask)
            / mask.size
        )

        qa_before = detect_chest_wall_side(mask)

        (
            image,
            mask,
            crop_box,
            crop_fractions
        ) = conservative_crop(
            image,
            mask,
            laterality
        )

        cropped_height, cropped_width = image.shape

        foreground_after = (
            np.count_nonzero(mask)
            / mask.size
        )

        (
            image,
            mask,
            expected_raw_side,
            flipped
        ) = apply_metadata_orientation(
            image,
            mask,
            laterality,
            laterality_mapping
        )

        qa_after = detect_chest_wall_side(mask)

        (
            image,
            scale,
            padding_x,
            padding_y,
            resized_width,
            resized_height
        ) = resize_and_pad(
            image,
            TARGET_SIZE
        )

        split = str(row["split"])

        output_directory = (
            OUTPUT_ROOT / split
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            output_directory
            / row["output_filename"]
        )

        Image.fromarray(
            image,
            mode="L"
        ).save(
            output_path,
            format="JPEG",
            quality=JPEG_QUALITY,
            subsampling=0
        )

        (
            x_min,
            y_min,
            x_max,
            y_max
        ) = crop_box

        total_removed_fraction = (
            1.0
            - (
                (cropped_width * cropped_height)
                / (original_width * original_height)
            )
        )

        df.at[index, "preprocessed_path"] = str(output_path)
        df.at[index, "preprocessing_status"] = "success"

        df.at[index, "laterality"] = laterality
        df.at[index, "orientation_flipped"] = flipped
        df.at[index, "expected_raw_chest_wall"] = expected_raw_side
        df.at[index, "qa_detected_chest_wall_before"] = qa_before
        df.at[index, "qa_detected_chest_wall_after"] = qa_after

        df.at[index, "original_width"] = original_width
        df.at[index, "original_height"] = original_height
        df.at[index, "cropped_width"] = cropped_width
        df.at[index, "cropped_height"] = cropped_height

        df.at[index, "crop_x_min"] = x_min
        df.at[index, "crop_y_min"] = y_min
        df.at[index, "crop_x_max"] = x_max
        df.at[index, "crop_y_max"] = y_max

        for name, value in crop_fractions.items():
            df.at[index, name] = value

        df.at[
            index,
            "total_removed_fraction"
        ] = total_removed_fraction

        df.at[index, "resize_scale"] = scale
        df.at[index, "resized_width"] = resized_width
        df.at[index, "resized_height"] = resized_height
        df.at[index, "padding_x"] = padding_x
        df.at[index, "padding_y"] = padding_y

        df.at[
            index,
            "foreground_fraction_before_crop"
        ] = foreground_before

        df.at[
            index,
            "foreground_fraction_after_crop"
        ] = foreground_after

    except Exception as exc:

        df.at[
            index,
            "preprocessing_status"
        ] = f"error: {exc}"

OUTPUT_MANIFEST.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_MANIFEST,
    index=False
)

print("\n")
print("=" * 72)
print("PREPROCESSING V3 SUMMARY")
print("=" * 72)

print("\nStatus:")
print(
    df["preprocessing_status"]
    .value_counts(dropna=False)
)

successful = df[
    df["preprocessing_status"] == "success"
].copy()

print("\nSuccessfully processed:")
print(
    f"{len(successful):,} / {len(df):,}"
)

print("\nMetadata-driven flips:")
print(
    successful[
        "orientation_flipped"
    ].value_counts()
)

print("\nTotal image area removed:")
print(
    pd.to_numeric(
        successful["total_removed_fraction"]
    ).describe()
)

print("\nMaximum side-specific crop fractions:")

for column in [
    "crop_left_fraction",
    "crop_right_fraction",
    "crop_top_fraction",
    "crop_bottom_fraction",
]:
    values = pd.to_numeric(successful[column])

    print(
        f"{column:24s} "
        f"mean={values.mean():.4f}, "
        f"max={values.max():.4f}"
    )

print("\nImages with >40% total area removed:")
print(
    (
        pd.to_numeric(
            successful["total_removed_fraction"]
        ) > 0.40
    ).sum()
)

print("\nForeground fraction BEFORE crop:")
print(
    pd.to_numeric(
        successful[
            "foreground_fraction_before_crop"
        ]
    ).describe()
)

print("\nForeground fraction AFTER crop:")
print(
    pd.to_numeric(
        successful[
            "foreground_fraction_after_crop"
        ]
    ).describe()
)

print("\nOutput images:")
print(
    successful["split"].value_counts()
)

print("\nSaved manifest:")
print(OUTPUT_MANIFEST.resolve())

print("\nOutput root:")
print(OUTPUT_ROOT.resolve())
