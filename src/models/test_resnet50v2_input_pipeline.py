from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path.cwd()

MANIFEST_FILE = Path(
    "data/processed/cbis_ddsm/manifests/"
    "cbis_ddsm_full_image_manifest_preprocessed_v4.csv"
)

OUTPUT_DIR = Path(
    "results/figures/augmentation"
)

IMAGE_SIZE = 512
BATCH_SIZE = 4
RANDOM_SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

tf.keras.utils.set_random_seed(RANDOM_SEED)

df = pd.read_csv(MANIFEST_FILE)

required_columns = [
    "preprocessed_path",
    "preprocessing_status",
    "binary_pathology",
    "split",
    "patient_id",
    "image_id",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Manifest is missing required columns: "
        + ", ".join(missing_columns)
    )

df = df[
    df["preprocessing_status"] == "success"
].copy()


def resolve_path(value):
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


df["resolved_path"] = (
    df["preprocessed_path"]
    .apply(resolve_path)
)

exists = (
    df["resolved_path"]
    .apply(os.path.exists)
)

print("=" * 72)
print("TENSORFLOW INPUT PIPELINE TEST")
print("=" * 72)

print(f"\nManifest rows after filtering: {len(df):,}")
print(f"Source images found: {exists.sum():,} / {len(exists):,}")

if not exists.all():
    missing = df.loc[
        ~exists,
        "resolved_path"
    ]

    print("\nFirst missing images:")
    for path in missing.head(10):
        print(path)

    raise FileNotFoundError(
        f"{(~exists).sum():,} preprocessed images could not be found."
    )


label_map = {
    "BENIGN": 0,
    "MALIGNANT": 1,
}

unknown_labels = sorted(
    set(
        df["binary_pathology"]
        .astype(str)
        .str.upper()
    )
    - set(label_map)
)

if unknown_labels:
    raise ValueError(
        "Unexpected pathology labels: "
        + ", ".join(unknown_labels)
    )

df["label"] = (
    df["binary_pathology"]
    .astype(str)
    .str.upper()
    .map(label_map)
    .astype(np.float32)
)

print("\nSplit counts:")
print(df["split"].value_counts())

print("\nClass counts by split:")
print(
    pd.crosstab(
        df["split"],
        df["binary_pathology"]
    )
)

train_df = df[
    df["split"] == "train"
].copy()

validation_df = df[
    df["split"] == "validation"
].copy()

test_df = df[
    df["split"] == "test"
].copy()


def load_image(path, label):
    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_jpeg(
        image_bytes,
        channels=1
    )

    image = tf.image.convert_image_dtype(
        image,
        tf.float32
    )

    image = image * 255.0

    image = tf.ensure_shape(
        image,
        [IMAGE_SIZE, IMAGE_SIZE, 1]
    )

    image = tf.image.grayscale_to_rgb(
        image
    )

    image = tf.ensure_shape(
        image,
        [IMAGE_SIZE, IMAGE_SIZE, 3]
    )

    return image, label


augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(
            factor=7.0 / 360.0,
            fill_mode="constant",
            fill_value=0.0,
            seed=RANDOM_SEED,
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05,
            fill_mode="constant",
            fill_value=0.0,
            seed=RANDOM_SEED + 1,
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.08, 0.08),
            width_factor=(-0.08, 0.08),
            fill_mode="constant",
            fill_value=0.0,
            seed=RANDOM_SEED + 2,
        ),
        tf.keras.layers.RandomContrast(
            factor=0.08,
            seed=RANDOM_SEED + 3,
        ),
    ],
    name="training_augmentation",
)

augmentation.build(
    (None, IMAGE_SIZE, IMAGE_SIZE, 3)
)

def augment_image(image, label):
    """
    Apply augmentation using an explicit batch dimension.

    The augmentation model therefore always receives input with
    shape (batch, height, width, channels).
    """

    image = tf.expand_dims(
        image,
        axis=0
    )

    image = augmentation(
        image,
        training=True
    )

    image = tf.squeeze(
        image,
        axis=0
    )

    image = tf.clip_by_value(
        image,
        0.0,
        255.0
    )

    return image, label

def resnet_preprocess(image, label):
    image = (
        tf.keras.applications.resnet_v2
        .preprocess_input(image)
    )

    return image, label


def build_dataset(
    frame,
    training=False,
    apply_resnet_preprocessing=True
):
    paths = frame[
        "resolved_path"
    ].to_numpy()

    labels = frame[
        "label"
    ].to_numpy(
        dtype=np.float32
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(frame),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    if training:
        dataset = dataset.map(
            augment_image,
            num_parallel_calls=AUTOTUNE
        )

    if apply_resnet_preprocessing:
        dataset = dataset.map(
            resnet_preprocess,
            num_parallel_calls=AUTOTUNE
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        AUTOTUNE
    )

    return dataset


train_dataset = build_dataset(
    train_df,
    training=True
)

validation_dataset = build_dataset(
    validation_df,
    training=False
)

test_dataset = build_dataset(
    test_df,
    training=False
)


def inspect_batch(dataset, name):
    images, labels = next(iter(dataset))

    print(f"\n{name} batch:")
    print("  image shape:", images.shape)
    print("  label shape:", labels.shape)
    print("  image dtype:", images.dtype)
    print("  label dtype:", labels.dtype)
    print(
        "  pixel range:",
        f"{tf.reduce_min(images).numpy():.3f}",
        "to",
        f"{tf.reduce_max(images).numpy():.3f}"
    )
    print("  labels:", labels.numpy())


inspect_batch(
    train_dataset,
    "Training"
)

inspect_batch(
    validation_dataset,
    "Validation"
)

inspect_batch(
    test_dataset,
    "Test"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

display_source = build_dataset(
    train_df,
    training=False,
    apply_resnet_preprocessing=False
)

source_images, source_labels = next(
    iter(display_source)
)

source_image = source_images[0]
source_label = int(
    source_labels[0].numpy()
)

label_name = (
    "MALIGNANT"
    if source_label == 1
    else "BENIGN"
)

fig, axes = plt.subplots(
    3,
    3,
    figsize=(10, 10)
)

axes = axes.flatten()

axes[0].imshow(
    source_image[..., 0].numpy(),
    cmap="gray",
    vmin=0,
    vmax=255
)
axes[0].set_title(
    f"Original ({label_name})"
)
axes[0].axis("off")

for index in range(1, 9):
    augmented = augmentation(
        tf.expand_dims(
            source_image,
            axis=0
        ),
        training=True
    )[0]

    augmented = tf.clip_by_value(
        augmented,
        0.0,
        255.0
    )

    axes[index].imshow(
        augmented[..., 0].numpy(),
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[index].set_title(
        f"Augmented {index}"
    )

    axes[index].axis("off")


fig.suptitle(
    "Training-Only Mammogram Augmentation",
    fontsize=14
)

fig.tight_layout()

augmentation_figure = (
    OUTPUT_DIR
    / "augmentation_examples.png"
)

fig.savefig(
    augmentation_figure,
    dpi=180,
    bbox_inches="tight"
)

plt.close(fig)

print(
    "\nAugmentation is applied only to the training dataset."
)

print(
    "Validation and test datasets use deterministic "
    "ResNet50V2 preprocessing only."
)

print("\nSaved augmentation figure:")
print(augmentation_figure.resolve())

print("\nInput pipeline test completed successfully.")
