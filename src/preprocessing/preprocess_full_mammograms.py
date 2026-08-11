from pathlib import Path
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
    "data/processed/cbis_ddsm/model_input/full_mammograms"
)

OUTPUT_MANIFEST = Path(
    "data/processed/cbis_ddsm/manifests/"
    "cbis_ddsm_full_image_manifest_preprocessed.csv"
)

TARGET_SIZE = 512

# Small amount of breast/background margin kept after cropping.
CROP_MARGIN = 0.02

JPEG_QUALITY = 95


# ============================================================
# Utility functions
# ============================================================

def load_grayscale(path):
    """
    Load a mammogram as an 8-bit grayscale NumPy array.
    """

    with Image.open(path) as image:
        image = image.convert("L")
        return np.asarray(image)


def create_breast_mask(image):
    """
    Estimate the breast region.

    CBIS-DDSM converted mammograms normally have a dark
    background. Otsu thresholding is used to separate the
    breast from the background, followed by morphological
    cleaning and selection of the largest connected component.
    """

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # --------------------------------------------------------
    # Otsu threshold
    # --------------------------------------------------------

    _, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --------------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Largest connected component
    # --------------------------------------------------------

    number_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    # label 0 = background
    if number_labels <= 1:
        raise ValueError(
            "No breast foreground component was detected."
        )

    component_areas = stats[
        1:,
        cv2.CC_STAT_AREA
    ]

    largest_label = (
        np.argmax(component_areas) + 1
    )

    mask = (
        labels == largest_label
    ).astype(np.uint8)

    return mask


def crop_to_breast(image, mask, margin_fraction=0.08):
    """
    Crop the image to the bounding box of the breast mask.

    A small margin is kept around the detected breast.
    """

    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        raise ValueError(
            "Breast mask contains no foreground."
        )

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    height, width = image.shape

    box_width = x_max - x_min + 1
    box_height = y_max - y_min + 1

    margin_x = int(
        box_width * margin_fraction
    )

    margin_y = int(
        box_height * margin_fraction
    )

    x_min = max(0, x_min - margin_x)
    x_max = min(
        width - 1,
        x_max + margin_x
    )

    y_min = max(0, y_min - margin_y)
    y_max = min(
        height - 1,
        y_max + margin_y
    )

    cropped_image = image[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    cropped_mask = mask[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    return (
        cropped_image,
        cropped_mask,
        (x_min, y_min, x_max, y_max)
    )


def detect_chest_wall_side(mask):
    """
    Estimate which side of the image contains the chest wall.

    Mammograms normally contain more breast tissue close to
    the chest-wall edge and taper toward the nipple.

    Tissue amounts in the left and right 20% of the image
    are compared.

    Returns:
        "left" or "right"
    """

    height, width = mask.shape

    edge_width = max(
        1,
        int(width * 0.20)
    )

    left_pixels = np.count_nonzero(
        mask[:, :edge_width]
    )

    right_pixels = np.count_nonzero(
        mask[:, -edge_width:]
    )

    if left_pixels >= right_pixels:
        return "left"

    return "right"


def normalise_orientation(image, mask):
    """
    Normalise all mammograms so that the chest wall is on
    the LEFT side of the image.

    Images whose chest wall is detected on the right are
    horizontally flipped.
    """

    chest_wall_side = detect_chest_wall_side(
        mask
    )

    flipped = False

    if chest_wall_side == "right":

        image = np.fliplr(image)
        mask = np.fliplr(mask)

        flipped = True

    return image, mask, chest_wall_side, flipped


def resize_and_pad(image, target_size=512):
    """
    Resize while preserving aspect ratio and pad the remaining
    space with black pixels.

    No stretching is performed.
    """

    original_height, original_width = (
        image.shape
    )

    scale = min(
        target_size / original_width,
        target_size / original_height
    )

    new_width = max(
        1,
        int(round(original_width * scale))
    )

    new_height = max(
        1,
        int(round(original_height * scale))
    )

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros(
        (target_size, target_size),
        dtype=np.uint8
    )

    x_offset = (
        target_size - new_width
    ) // 2

    y_offset = (
        target_size - new_height
    ) // 2

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


def build_output_filename(row):

    patient = str(row["patient_id"]).strip()
    breast = str(row["left_or_right_breast"]).strip()
    view = str(row["image_view"]).strip()

    image_id = str(
        row["image_id"]
    )[:8]

    return (
        f"{patient}_{breast}_{view}_"
        f"{image_id}.jpg"
    )


# ============================================================
# Load manifest
# ============================================================

df = pd.read_csv(
    MANIFEST_FILE
)

print("=" * 72)
print("FULL-MAMMOGRAM PREPROCESSING")
print("=" * 72)

print(f"\nManifest rows: {len(df):,}")

print("\nSplits:")
print(
    df["split"].value_counts()
)

def create_image_id(path_value):
    return Path(
        str(path_value)
    ).stem


df["image_id"] = (
    df["full_mammogram_path"]
    .apply(create_image_id)
)

print(
    "\nUnique image IDs:",
    df["image_id"].nunique(),
    "/",
    len(df)
)

if df["image_id"].nunique() != len(df):
    raise ValueError(
        "image_id is not unique."
    )
# ============================================================
# Verify source paths before preprocessing
# ============================================================

def resolve_manifest_path(value):
    path = Path(str(value))

    if not path.is_absolute():
        path = Path.cwd() / path

    return path


source_paths = df["full_jpeg_path"].apply(
    resolve_manifest_path
)

existing = source_paths.apply(
    lambda p: p.exists()
)

print("\nSource image verification:")
print(
    f"Found: {existing.sum():,} / {len(existing):,}"
)


if not existing.all():

    missing = source_paths[~existing]

    print("\nFirst missing paths:")

    for path in missing.head(10):
        print(path)

    raise FileNotFoundError(
        f"{(~existing).sum():,} source mammograms "
        "could not be found. Preprocessing stopped."
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
    "chest_wall_original",
    "orientation_flipped",
    "resize_scale",
    "resized_width",
    "resized_height",
    "padding_x",
    "padding_y",
    "foreground_fraction",
]

for column in output_columns:
    if column not in df.columns:
        df[column] = None

df["output_filename"] = df.apply(
    build_output_filename,
    axis=1
)

duplicate_names = (
    df["output_filename"]
    .duplicated(keep=False)
)

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
            ]
        ].to_string(index=False)
    )

    raise ValueError(
        "Output filenames are not unique."
    )

print(
    "\nUnique output filenames:",
    df["output_filename"].nunique(),
    "/",
    len(df)
)

# ============================================================
# Preprocess
# ============================================================

for index, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Preprocessing mammograms"
):

    try:

        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------

        source_path = resolve_manifest_path(
            row["full_jpeg_path"]
        )

        image = load_grayscale(
            source_path
        )

        original_height, original_width = (
            image.shape
        )

        # ----------------------------------------------------
        # Breast segmentation / background removal
        # ----------------------------------------------------

        mask = create_breast_mask(
            image
        )

        foreground_fraction = (
            np.count_nonzero(mask)
            / mask.size
        )

        (
            image,
            mask,
            crop_box
        ) = crop_to_breast(
            image,
            mask,
            CROP_MARGIN
        )

        cropped_height, cropped_width = (
            image.shape
        )

        # ----------------------------------------------------
        # Orientation normalisation
        # ----------------------------------------------------

        (
            image,
            mask,
            chest_wall_side,
            flipped
        ) = normalise_orientation(
            image,
            mask
        )

        # ----------------------------------------------------
        # Aspect-preserving resize
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Output path
        # ----------------------------------------------------

        split = str(
            row["split"]
        )

        output_directory = (
            OUTPUT_ROOT / split
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        filename = row["output_filename"]

        output_path = (
            output_directory / filename
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        Image.fromarray(
            image,
            mode="L"
        ).save(
            output_path,
            format="JPEG",
            quality=JPEG_QUALITY,
            subsampling=0
        )

        # ----------------------------------------------------
        # Record preprocessing metadata
        # ----------------------------------------------------

        (
            x_min,
            y_min,
            x_max,
            y_max
        ) = crop_box

        df.at[
            index,
            "preprocessed_path"
        ] = str(
            output_path.resolve()
        )

        df.at[
            index,
            "preprocessing_status"
        ] = "success"

        df.at[
            index,
            "original_width"
        ] = original_width

        df.at[
            index,
            "original_height"
        ] = original_height

        df.at[
            index,
            "cropped_width"
        ] = cropped_width

        df.at[
            index,
            "cropped_height"
        ] = cropped_height

        df.at[index, "crop_x_min"] = x_min
        df.at[index, "crop_y_min"] = y_min
        df.at[index, "crop_x_max"] = x_max
        df.at[index, "crop_y_max"] = y_max

        df.at[
            index,
            "chest_wall_original"
        ] = chest_wall_side

        df.at[
            index,
            "orientation_flipped"
        ] = flipped

        df.at[
            index,
            "resize_scale"
        ] = scale

        df.at[
            index,
            "resized_width"
        ] = resized_width

        df.at[
            index,
            "resized_height"
        ] = resized_height

        df.at[
            index,
            "padding_x"
        ] = padding_x

        df.at[
            index,
            "padding_y"
        ] = padding_y

        df.at[
            index,
            "foreground_fraction"
        ] = foreground_fraction

    except Exception as exc:

        df.at[
            index,
            "preprocessing_status"
        ] = f"error: {exc}"


# ============================================================
# Save updated manifest
# ============================================================

OUTPUT_MANIFEST.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_MANIFEST,
    index=False
)


# ============================================================
# Summary
# ============================================================

print("\n")
print("=" * 72)
print("PREPROCESSING SUMMARY")
print("=" * 72)

print("\nStatus:")
print(
    df["preprocessing_status"]
    .value_counts(dropna=False)
)

successful = df[
    df["preprocessing_status"] == "success"
]

print("\nSuccessfully processed:")
print(
    f"{len(successful):,} / {len(df):,}"
)

print("\nOrientation originally detected:")
print(
    successful[
        "chest_wall_original"
    ].value_counts()
)

print("\nImages horizontally flipped:")
print(
    successful[
        "orientation_flipped"
    ].value_counts()
)

print("\nOriginal image dimensions:")
print(
    successful[
        [
            "original_width",
            "original_height"
        ]
    ]
    .describe()
)

print("\nCropped image dimensions:")
print(
    successful[
        [
            "cropped_width",
            "cropped_height"
        ]
    ]
    .describe()
)

print("\nForeground fraction:")
print(
    successful[
        "foreground_fraction"
    ].describe()
)

print("\nOutput images:")
print(
    successful["split"]
    .value_counts()
)

print("\nSaved manifest:")
print(
    OUTPUT_MANIFEST.resolve()
)