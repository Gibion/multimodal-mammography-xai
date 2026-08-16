from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path.cwd()

MANIFEST_FILE = Path(
    "data/processed/cbis_ddsm/manifests/"
    "cbis_ddsm_full_image_manifest_preprocessed_v4.csv"
)

MODEL_PATH = Path(
    "models/resnet50v2_finetuned/best_model.keras"
)

EVALUATION_PREDICTIONS = Path(
    "results/resnet50v2_finetuned/evaluation/"
    "test_predictions.csv"
)

OUTPUT_DIR = Path(
    "results/resnet50v2_finetuned/gradcam"
)

FIGURES_DIR = OUTPUT_DIR / "figures"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
OVERLAY_DIR = OUTPUT_DIR / "overlays"

IMAGE_SIZE = 512
THRESHOLD = 0.5
EXAMPLES_PER_CATEGORY = 5
OVERLAY_ALPHA = 0.40


# ============================================================
# Output directories
# ============================================================

for directory in [
    OUTPUT_DIR,
    FIGURES_DIR,
    HEATMAP_DIR,
    OVERLAY_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# Helpers
# ============================================================

def resolve_path(value):
    path = Path(str(value))

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_original_grayscale(path):
    with Image.open(path) as image:
        return np.asarray(
            image.convert("L")
        )


def load_model_input(path):
    image_bytes = tf.io.read_file(
        str(path)
    )

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

    image = (
        tf.keras.applications.resnet_v2
        .preprocess_input(image)
    )

    return tf.expand_dims(
        image,
        axis=0
    )


def locate_resnet_backbone(model):
    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.Model
        ) and "resnet50v2" in layer.name.lower():
            return layer

    raise ValueError(
        "Could not find the ResNet50V2 backbone "
        "inside the loaded model."
    )


def locate_head_layers(model):
    """
    Locate the classifier head from the saved baseline/fine-tuned model.

    Expected architecture:
        ResNet50V2
        -> GlobalAveragePooling2D
        -> Dense(256, relu)
        -> Dropout
        -> Dense(1, sigmoid)
    """

    layer_names = [
        "global_average_pooling",
        "dense_256",
        "dropout",
        "prediction",
    ]

    layers = {}

    for name in layer_names:
        try:
            layers[name] = model.get_layer(name)
        except ValueError as exc:
            raise ValueError(
                f"Expected classifier-head layer '{name}' "
                "was not found in the loaded model."
            ) from exc

    return layers


def forward_from_feature_maps(
    feature_maps,
    head_layers
):
    """
    Run the classifier head directly from the ResNet feature maps.

    This avoids the Keras 3 nested-model graph connectivity problem
    caused by trying to combine backbone.output with model.inputs
    inside a new Functional model.
    """

    x = head_layers[
        "global_average_pooling"
    ](
        feature_maps
    )

    x = head_layers[
        "dense_256"
    ](
        x
    )

    # Dropout must be disabled for deterministic inference/Grad-CAM.
    x = head_layers[
        "dropout"
    ](
        x,
        training=False
    )

    prediction = head_layers[
        "prediction"
    ](
        x
    )

    return prediction


def generate_gradcam(
    image_tensor,
    backbone,
    head_layers
):
    """
    Compute Grad-CAM for the malignant sigmoid output.

    The gradient is taken with respect to the final convolutional
    feature map returned by ResNet50V2(include_top=False).
    """

    with tf.GradientTape() as tape:

        feature_maps = backbone(
            image_tensor,
            training=False
        )

        # Explicitly watch the feature map because it is the tensor
        # with respect to which we need gradients.
        tape.watch(
            feature_maps
        )

        prediction = forward_from_feature_maps(
            feature_maps,
            head_layers
        )

        malignant_score = prediction[
            :,
            0
        ]

    gradients = tape.gradient(
        malignant_score,
        feature_maps
    )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients are None. "
            "Check the backbone/head connection."
        )

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    heatmap = tf.reduce_sum(
        feature_maps
        * weights[
            :,
            tf.newaxis,
            tf.newaxis,
            :
        ],
        axis=-1
    )

    heatmap = tf.nn.relu(
        heatmap
    )

    heatmap = heatmap[0]

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = tf.cond(
        max_value > 0,
        lambda: heatmap / max_value,
        lambda: tf.zeros_like(heatmap)
    )

    heatmap = tf.image.resize(
        heatmap[
            ...,
            tf.newaxis
        ],
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        method="bilinear"
    )

    heatmap = tf.squeeze(
        heatmap,
        axis=-1
    )

    return (
        heatmap.numpy(),
        float(
            prediction[
                0,
                0
            ].numpy()
        )
    )


def save_heatmap_image(
    heatmap,
    output_path
):
    fig = plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        heatmap,
        vmin=0.0,
        vmax=1.0
    )

    plt.axis(
        "off"
    )

    plt.tight_layout(
        pad=0
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close(
        fig
    )


def save_overlay_image(
    original,
    heatmap,
    output_path
):
    fig = plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        original,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    plt.imshow(
        heatmap,
        alpha=OVERLAY_ALPHA,
        vmin=0.0,
        vmax=1.0
    )

    plt.axis(
        "off"
    )

    plt.tight_layout(
        pad=0
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close(
        fig
    )


def outcome_category(
    true_label,
    predicted_label
):
    if true_label == 1 and predicted_label == 1:
        return "TP"

    if true_label == 0 and predicted_label == 0:
        return "TN"

    if true_label == 0 and predicted_label == 1:
        return "FP"

    return "FN"


# ============================================================
# Load model
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Fine-tuned model not found: "
        f"{MODEL_PATH.resolve()}"
    )

print("=" * 72)
print("GRAD-CAM GENERATION FOR FINE-TUNED RESNET50V2")
print("=" * 72)

print(
    "\nLoading model:"
)

print(
    MODEL_PATH.resolve()
)

model = tf.keras.models.load_model(
    MODEL_PATH
)

backbone = locate_resnet_backbone(
    model
)

head_layers = locate_head_layers(
    model
)

print(
    "\nBackbone:"
)

print(
    backbone.name
)

print(
    "\nBackbone output shape:"
)

print(
    backbone.output_shape
)

print(
    "\nClassifier head:"
)

for name, layer in head_layers.items():
    print(
        f"  {name:24s} "
        f"{layer.__class__.__name__}"
    )


# ============================================================
# Sanity check
# ============================================================

# Verify that manually passing the backbone features through the
# classifier head reproduces the saved model's prediction.

manifest = pd.read_csv(
    MANIFEST_FILE
)

sample_row = manifest[
    (
        manifest[
            "preprocessing_status"
        ] == "success"
    )
    &
    (
        manifest[
            "split"
        ] == "test"
    )
].iloc[0]

sample_tensor = load_model_input(
    resolve_path(
        sample_row[
            "preprocessed_path"
        ]
    )
)

full_prediction = float(
    model(
        sample_tensor,
        training=False
    )[
        0,
        0
    ].numpy()
)

sample_features = backbone(
    sample_tensor,
    training=False
)

manual_prediction = float(
    forward_from_feature_maps(
        sample_features,
        head_layers
    )[
        0,
        0
    ].numpy()
)

prediction_difference = abs(
    full_prediction
    - manual_prediction
)

print(
    "\nGrad-CAM forward-pass sanity check:"
)

print(
    f"Full model prediction:   "
    f"{full_prediction:.8f}"
)

print(
    f"Manual head prediction:  "
    f"{manual_prediction:.8f}"
)

print(
    f"Absolute difference:      "
    f"{prediction_difference:.10f}"
)

if prediction_difference > 1e-5:
    raise RuntimeError(
        "Manual classifier-head forward pass does not match "
        "the saved model prediction closely enough."
    )


# ============================================================
# Load test predictions
# ============================================================

if EVALUATION_PREDICTIONS.exists():

    print(
        "\nUsing saved B1 test predictions:"
    )

    print(
        EVALUATION_PREDICTIONS.resolve()
    )

    test_df = pd.read_csv(
        EVALUATION_PREDICTIONS
    )

else:

    raise FileNotFoundError(
        "Expected evaluation prediction file was not found: "
        f"{EVALUATION_PREDICTIONS.resolve()}\n"
        "Run evaluate_resnet50v2_finetuned.py first."
    )


required_prediction_columns = [
    "image_id",
    "patient_id",
    "binary_pathology",
    "label",
    "preprocessed_path",
    "predicted_probability",
]

missing_columns = [
    column
    for column in required_prediction_columns
    if column not in test_df.columns
]

if missing_columns:
    raise ValueError(
        "Test prediction file is missing columns: "
        + ", ".join(
            missing_columns
        )
    )


# ============================================================
# Outcome categories at threshold 0.5
# ============================================================

test_df[
    "predicted_label"
] = (
    test_df[
        "predicted_probability"
    ]
    >= THRESHOLD
).astype(int)

test_df[
    "outcome"
] = test_df.apply(
    lambda row: outcome_category(
        int(
            row[
                "label"
            ]
        ),
        int(
            row[
                "predicted_label"
            ]
        )
    ),
    axis=1
)

test_df[
    "classification_confidence"
] = np.where(
    test_df[
        "predicted_label"
    ] == 1,
    test_df[
        "predicted_probability"
    ],
    1.0
    - test_df[
        "predicted_probability"
    ]
)

print(
    "\nTest outcome counts at threshold 0.5:"
)

print(
    test_df[
        "outcome"
    ].value_counts()
)


# ============================================================
# Select high-confidence examples
# ============================================================

selected_frames = []

for category in [
    "TP",
    "TN",
    "FP",
    "FN",
]:

    category_df = test_df[
        test_df[
            "outcome"
        ] == category
    ].copy()

    category_df = (
        category_df
        .sort_values(
            "classification_confidence",
            ascending=False
        )
        .head(
            EXAMPLES_PER_CATEGORY
        )
    )

    selected_frames.append(
        category_df
    )

selected_df = pd.concat(
    selected_frames,
    ignore_index=True
)

print(
    "\nSelected Grad-CAM examples:"
)

print(
    selected_df[
        [
            "patient_id",
            "binary_pathology",
            "predicted_probability",
            "outcome",
            "classification_confidence",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Generate Grad-CAM outputs
# ============================================================

records = []

print(
    "\nGenerating Grad-CAM maps..."
)

for _, row in selected_df.iterrows():

    image_path = resolve_path(
        row[
            "preprocessed_path"
        ]
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: "
            f"{image_path}"
        )

    original = load_original_grayscale(
        image_path
    )

    image_tensor = load_model_input(
        image_path
    )

    heatmap, model_probability = (
        generate_gradcam(
            image_tensor,
            backbone,
            head_layers
        )
    )

    safe_id = (
        str(
            row[
                "image_id"
            ]
        )
        .replace(
            "/",
            "_"
        )
        .replace(
            "\\",
            "_"
        )
    )

    prefix = (
        f"{row['outcome']}_"
        f"{row['patient_id']}_"
        f"{safe_id}"
    )

    heatmap_path = (
        HEATMAP_DIR
        / f"{prefix}_heatmap.png"
    )

    overlay_path = (
        OVERLAY_DIR
        / f"{prefix}_overlay.png"
    )

    save_heatmap_image(
        heatmap,
        heatmap_path
    )

    save_overlay_image(
        original,
        heatmap,
        overlay_path
    )

    max_y, max_x = np.unravel_index(
        np.argmax(
            heatmap
        ),
        heatmap.shape
    )

    saved_probability = float(
        row[
            "predicted_probability"
        ]
    )

    records.append({
        "image_id": row[
            "image_id"
        ],
        "patient_id": row[
            "patient_id"
        ],
        "binary_pathology": row[
            "binary_pathology"
        ],
        "true_label": int(
            row[
                "label"
            ]
        ),
        "predicted_label": int(
            row[
                "predicted_label"
            ]
        ),
        "outcome": row[
            "outcome"
        ],
        "saved_probability": saved_probability,
        "gradcam_probability": float(
            model_probability
        ),
        "probability_difference": abs(
            saved_probability
            - model_probability
        ),
        "classification_confidence": float(
            row[
                "classification_confidence"
            ]
        ),
        "gradcam_peak_x": int(
            max_x
        ),
        "gradcam_peak_y": int(
            max_y
        ),
        "preprocessed_path": str(
            image_path
        ),
        "heatmap_path": str(
            heatmap_path.resolve()
        ),
        "overlay_path": str(
            overlay_path.resolve()
        ),
    })


gradcam_manifest = pd.DataFrame(
    records
)

gradcam_manifest.to_csv(
    OUTPUT_DIR
    / "gradcam_examples_manifest.csv",
    index=False
)


# ============================================================
# Create comparison figure
# ============================================================

category_order = [
    "TP",
    "TN",
    "FP",
    "FN",
]

n_columns = EXAMPLES_PER_CATEGORY

fig, axes = plt.subplots(
    len(
        category_order
    ),
    n_columns,
    figsize=(
        3.2
        * n_columns,
        3.5
        * len(
            category_order
        )
    )
)

if n_columns == 1:
    axes = np.asarray(
        axes
    )[
        :,
        np.newaxis
    ]

for row_index, category in enumerate(
    category_order
):

    category_records = (
        gradcam_manifest[
            gradcam_manifest[
                "outcome"
            ] == category
        ]
        .sort_values(
            "classification_confidence",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    for column_index in range(
        n_columns
    ):

        ax = axes[
            row_index,
            column_index
        ]

        ax.axis(
            "off"
        )

        if column_index >= len(
            category_records
        ):
            continue

        record = category_records.iloc[
            column_index
        ]

        original = load_original_grayscale(
            Path(
                record[
                    "preprocessed_path"
                ]
            )
        )

        image_tensor = load_model_input(
            Path(
                record[
                    "preprocessed_path"
                ]
            )
        )

        heatmap, _ = generate_gradcam(
            image_tensor,
            backbone,
            head_layers
        )

        ax.imshow(
            original,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        ax.imshow(
            heatmap,
            alpha=OVERLAY_ALPHA,
            vmin=0.0,
            vmax=1.0
        )

        ax.set_title(
            f"{category} | "
            f"{record['patient_id']}\n"
            f"P(malignant)="
            f"{record['saved_probability']:.3f}",
            fontsize=9
        )


fig.suptitle(
    "Fine-Tuned ResNet50V2 Grad-CAM Examples",
    fontsize=16
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.97
    ]
)

comparison_figure = (
    FIGURES_DIR
    / "gradcam_tp_tn_fp_fn_examples.png"
)

fig.savefig(
    comparison_figure,
    dpi=180,
    bbox_inches="tight"
)

plt.close(
    fig
)


# ============================================================
# Final summary
# ============================================================

print("\n")
print("=" * 72)
print("GRAD-CAM GENERATION COMPLETE")
print("=" * 72)

print(
    f"\nGrad-CAM examples generated: "
    f"{len(gradcam_manifest)}"
)

print(
    "\nExamples by category:"
)

print(
    gradcam_manifest[
        "outcome"
    ].value_counts()
)

print(
    "\nMaximum prediction difference:"
)

print(
    f"{gradcam_manifest['probability_difference'].max():.10f}"
)

print(
    "\nSaved manifest:"
)

print(
    (
        OUTPUT_DIR
        / "gradcam_examples_manifest.csv"
    ).resolve()
)

print(
    "\nSaved comparison figure:"
)

print(
    comparison_figure.resolve()
)

print(
    "\nIndividual overlays:"
)

print(
    OVERLAY_DIR.resolve()
)
