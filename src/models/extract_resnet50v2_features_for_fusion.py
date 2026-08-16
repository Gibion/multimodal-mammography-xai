from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path.cwd()

MODEL_FILE = Path(
    "models/resnet50v2_finetuned/best_model.keras"
)

RADIOMICS_SELECTED_FILE = Path(
    "data/processed/cbis_ddsm/radiomics/feature_selection/"
    "cbis_ddsm_radiomics_selected_features.csv"
)

OUTPUT_DIR = Path(
    "data/processed/cbis_ddsm/fusion"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "cbis_ddsm_resnet50v2_dense256_features.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "cbis_ddsm_resnet50v2_dense256_features_summary.json"
)

BATCH_SIZE = 8

# The layer used for fusion.
FEATURE_LAYER_NAME = "dense_256"

# Set to an integer such as 20 for a quick smoke test.
# Leave as None for the full matched cohort.
MAX_CASES = None


# ============================================================
# Helpers
# ============================================================

def resolve_path(value):
    if pd.isna(value):
        return None

    path = Path(str(value))

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def label_to_int(value):
    text = str(value).strip().upper()

    if text == "MALIGNANT":
        return 1

    if text == "BENIGN":
        return 0

    raise ValueError(
        f"Unexpected binary_pathology value: {value}"
    )


def load_and_preprocess_image(path):
    """
    Load one V4 preprocessed mammogram.

    `path` may be either:
    - a TensorFlow string tensor when called from tf.data, or
    - a Python string when used directly.
    """

    image_bytes = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image_bytes,
        channels=3,
    )

    image = tf.image.convert_image_dtype(
        image,
        tf.float32,
    )

    image = tf.image.resize(
        image,
        (512, 512),
        method="bilinear",
    )

    image = tf.keras.applications.resnet_v2.preprocess_input(
        image * 255.0
    )

    return image


def build_dataset(frame):
    paths = [
        str(
            resolve_path(path)
        )
        for path in frame[
            "preprocessed_path"
        ]
    ]

    ds = tf.data.Dataset.from_tensor_slices(
        paths
    )

    ds = ds.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(
        BATCH_SIZE
    )

    ds = ds.prefetch(
        tf.data.AUTOTUNE
    )

    return ds


def find_feature_layer(model, requested_name):
    """
    Prefer the explicit dense_256 layer. If the loaded Keras model
    contains nested layers, search recursively.
    """

    try:
        return model.get_layer(
            requested_name
        )
    except ValueError:
        pass

    for layer in model.layers:
        if isinstance(
            layer,
            tf.keras.Model
        ):
            try:
                return layer.get_layer(
                    requested_name
                )
            except ValueError:
                continue

    raise ValueError(
        f"Could not find feature layer '{requested_name}'. "
        "Available top-level layers are:\n"
        + "\n".join(
            layer.name
            for layer in model.layers
        )
    )


# ============================================================
# Validate files
# ============================================================

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Fine-tuned model not found:\n"
        f"{MODEL_FILE.resolve()}"
    )

if not RADIOMICS_SELECTED_FILE.exists():
    raise FileNotFoundError(
        f"Radiomics matched cohort file not found:\n"
        f"{RADIOMICS_SELECTED_FILE.resolve()}"
    )


# ============================================================
# Load matched cohort
# ============================================================

df = pd.read_csv(
    RADIOMICS_SELECTED_FILE
)

required_columns = [
    "full_image_id",
    "patient_id",
    "split",
    "binary_pathology",
    "preprocessed_path",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Matched radiomics file is missing required columns: "
        + ", ".join(
            missing_columns
        )
    )

if df[
    "full_image_id"
].duplicated().any():
    raise ValueError(
        "Duplicate full_image_id rows found in matched cohort."
    )

if MAX_CASES is not None:
    df = df.head(
        int(
            MAX_CASES
        )
    ).copy()

df = df.reset_index(
    drop=True
)

df[
    "label"
] = df[
    "binary_pathology"
].apply(
    label_to_int
)

# Verify that all CNN input images exist.
df[
    "cnn_input_exists"
] = df[
    "preprocessed_path"
].apply(
    lambda value: (
        resolve_path(
            value
        ) is not None
        and resolve_path(
            value
        ).exists()
    )
)

if not df[
    "cnn_input_exists"
].all():
    missing = df[
        ~df[
            "cnn_input_exists"
        ]
    ]

    raise FileNotFoundError(
        "Some matched-cohort CNN input images are missing:\n"
        + missing[
            [
                "full_image_id",
                "patient_id",
                "preprocessed_path",
            ]
        ].head(
            20
        ).to_string(
            index=False
        )
    )


# ============================================================
# Load fine-tuned CNN
# ============================================================

print("=" * 72)
print("EXTRACT RESNET50V2 FEATURES FOR MULTIMODAL FUSION")
print("=" * 72)

print(
    "\nLoading fine-tuned model:"
)

print(
    MODEL_FILE.resolve()
)

model = tf.keras.models.load_model(
    MODEL_FILE
)

feature_layer = find_feature_layer(
    model,
    FEATURE_LAYER_NAME
)

print(
    f"\nFusion feature layer: "
    f"{feature_layer.name}"
)

print(
    f"Feature layer output shape: "
    f"{feature_layer.output.shape}"
)

feature_extractor = tf.keras.Model(
    inputs=model.input,
    outputs=feature_layer.output,
    name="resnet50v2_dense256_feature_extractor",
)


# ============================================================
# Forward-pass sanity check
# ============================================================

sample_path = resolve_path(
    df.iloc[0]["preprocessed_path"]
)

sample_image = load_and_preprocess_image(
    str(sample_path)
)

sample_batch = tf.expand_dims(
    sample_image,
    axis=0
)

full_prediction = model(
    sample_batch,
    training=False,
)

feature_vector = feature_extractor(
    sample_batch,
    training=False,
)

print(
    "\nForward-pass sanity check:"
)

print(
    f"Full model prediction shape: "
    f"{tuple(full_prediction.shape)}"
)

print(
    f"Feature vector shape: "
    f"{tuple(feature_vector.shape)}"
)

if len(
    feature_vector.shape
) != 2:
    raise ValueError(
        "Expected a 2-D feature matrix from dense_256."
    )

feature_dimension = int(
    feature_vector.shape[-1]
)

print(
    f"Feature dimension: "
    f"{feature_dimension}"
)


# ============================================================
# Extract split-by-split to preserve ordering
# ============================================================

all_outputs = []

start_time = time.time()

for split_name in [
    "train",
    "validation",
    "test",
]:
    split_df = df[
        df[
            "split"
        ] == split_name
    ].copy()

    if len(
        split_df
    ) == 0:
        continue

    print(
        f"\nExtracting {split_name} features: "
        f"{len(split_df):,} mammograms"
    )

    dataset = build_dataset(
        split_df
    )

    features = feature_extractor.predict(
        dataset,
        verbose=1,
    )

    if features.shape[
        0
    ] != len(
        split_df
    ):
        raise RuntimeError(
            f"{split_name}: extracted feature rows "
            f"{features.shape[0]} != expected {len(split_df)}."
        )

    if features.shape[
        1
    ] != feature_dimension:
        raise RuntimeError(
            f"{split_name}: unexpected feature dimension "
            f"{features.shape[1]}."
        )

    metadata_columns = [
        column
        for column in [
            "full_image_id",
            "patient_id",
            "split",
            "binary_pathology",
            "label",
            "preprocessed_path",
            "full_mammogram_path",
            "n_lesions",
            "radiomics_lesion_group",
        ]
        if column in split_df.columns
    ]

    output = split_df[
        metadata_columns
    ].reset_index(
        drop=True
    ).copy()

    feature_columns = [
        f"cnn_dense256_{index:03d}"
        for index in range(
            feature_dimension
        )
    ]

    feature_frame = pd.DataFrame(
        features,
        columns=feature_columns,
    )

    output = pd.concat(
        [
            output,
            feature_frame,
        ],
        axis=1,
    )

    all_outputs.append(
        output
    )


# ============================================================
# Combine and QA
# ============================================================

features_df = pd.concat(
    all_outputs,
    ignore_index=True,
)

feature_columns = [
    column
    for column in features_df.columns
    if column.startswith(
        "cnn_dense256_"
    )
]

feature_array = features_df[
    feature_columns
].to_numpy(
    dtype=float
)

n_nan = int(
    np.isnan(
        feature_array
    ).sum()
)

n_inf = int(
    np.isinf(
        feature_array
    ).sum()
)

constant_features = []

for column in feature_columns:
    if features_df[
        column
    ].nunique(
        dropna=True
    ) <= 1:
        constant_features.append(
            column
        )

if n_nan > 0:
    raise ValueError(
        f"Extracted CNN feature matrix contains "
        f"{n_nan} NaN values."
    )

if n_inf > 0:
    raise ValueError(
        f"Extracted CNN feature matrix contains "
        f"{n_inf} infinite values."
    )

if features_df[
    "full_image_id"
].duplicated().any():
    raise ValueError(
        "Duplicate full_image_id rows found after extraction."
    )


# ============================================================
# Save outputs
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

features_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

elapsed_seconds = (
    time.time()
    - start_time
)

summary = {
    "input_mammograms": int(
        len(
            df
        )
    ),
    "feature_layer": FEATURE_LAYER_NAME,
    "feature_dimension": int(
        feature_dimension
    ),
    "split_counts": {
        str(
            key
        ): int(
            value
        )
        for key, value
        in features_df[
            "split"
        ].value_counts().items()
    },
    "class_counts_by_split": {
        split_name: {
            str(
                key
            ): int(
                value
            )
            for key, value
            in features_df.loc[
                features_df[
                    "split"
                ] == split_name,
                "binary_pathology",
            ].value_counts().items()
        }
        for split_name in [
            "train",
            "validation",
            "test",
        ]
    },
    "nan_values": n_nan,
    "infinite_values": n_inf,
    "constant_feature_count": int(
        len(
            constant_features
        )
    ),
    "constant_features": constant_features,
    "elapsed_seconds": float(
        elapsed_seconds
    ),
}

with SUMMARY_FILE.open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        summary,
        handle,
        indent=2,
    )


# ============================================================
# Console summary
# ============================================================

print("\n")
print("=" * 72)
print("CNN FEATURE EXTRACTION COMPLETE")
print("=" * 72)

print(
    f"\nMammograms processed: "
    f"{len(features_df):,}"
)

print(
    f"CNN feature dimension: "
    f"{feature_dimension}"
)

print(
    "\nSplit counts:"
)

print(
    features_df[
        "split"
    ].value_counts()
)

print(
    "\nClass counts by split:"
)

print(
    pd.crosstab(
        features_df[
            "split"
        ],
        features_df[
            "binary_pathology"
        ],
    )
)

print(
    f"\nNaN values: "
    f"{n_nan}"
)

print(
    f"Infinite values: "
    f"{n_inf}"
)

print(
    f"Constant CNN features: "
    f"{len(constant_features)}"
)

print(
    f"\nElapsed time: "
    f"{elapsed_seconds / 60.0:.1f} minutes"
)

print(
    "\nSaved CNN fusion features:"
)

print(
    OUTPUT_FILE.resolve()
)

print(
    "\nSaved summary:"
)

print(
    SUMMARY_FILE.resolve()
)
