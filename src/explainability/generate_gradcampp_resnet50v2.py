"""
Generate Grad-CAM++ explanation maps for the fine-tuned ResNet50V2
classifier on the exact ROI-evaluable CBIS-DDSM test cohort.
"""

from pathlib import Path
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path.cwd()

MODEL_FILE = Path("models/resnet50v2_finetuned/best_model.keras")
TEST_PREDICTIONS_FILE = Path(
    "results/resnet50v2_finetuned/evaluation/test_predictions.csv"
)
ROI_XAI_MANIFEST_FILE = Path(
    "data/processed/cbis_ddsm/manifests/cbis_ddsm_roi_xai_manifest.csv"
)
EXISTING_GRADCAM_EXAMPLES_FILE = Path(
    "results/resnet50v2_finetuned/gradcam/gradcam_examples_manifest.csv"
)

OUTPUT_ROOT = Path("results/resnet50v2_finetuned/gradcampp")
HEATMAP_NPY_DIR = OUTPUT_ROOT / "heatmaps_npy"
HEATMAP_PNG_DIR = OUTPUT_ROOT / "heatmaps_png"
OVERLAY_DIR = OUTPUT_ROOT / "overlays"
FIGURE_DIR = OUTPUT_ROOT / "figures"

OUTPUT_MANIFEST = OUTPUT_ROOT / "gradcampp_test_manifest.csv"
SUMMARY_FILE = OUTPUT_ROOT / "gradcampp_summary.json"
COMPARISON_FIGURE = FIGURE_DIR / "gradcampp_tp_tn_fp_fn_examples.png"

BACKBONE_LAYER_NAME = "resnet50v2"
IMAGE_SIZE = (512, 512)
CLASSIFICATION_THRESHOLD = 0.5
EPSILON = 1e-10
OVERLAY_ALPHA = 0.45
MAX_CASES = None


def resolve_path(value):
    if pd.isna(value):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def find_column(frame, candidates, required=True):
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    if required:
        raise ValueError(
            "Could not find any of these columns: " + ", ".join(candidates)
        )
    return None


def pathology_to_int(value):
    text = str(value).strip().upper()
    if text == "MALIGNANT":
        return 1
    if text == "BENIGN":
        return 0
    if text in {"0", "1"}:
        return int(text)
    raise ValueError(f"Unexpected pathology label: {value}")


def classify_outcome(label, probability, threshold=0.5):
    prediction = int(probability >= threshold)
    if label == 1 and prediction == 1:
        return "TP"
    if label == 0 and prediction == 0:
        return "TN"
    if label == 0 and prediction == 1:
        return "FP"
    return "FN"


def classification_confidence(probability, outcome):
    if outcome in {"TP", "FP"}:
        return float(probability)
    return float(1.0 - probability)


def load_model_input_image(path):
    image_bytes = tf.io.read_file(str(path))
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")
    display_image = tf.clip_by_value(image, 0.0, 1.0)
    model_input = tf.keras.applications.resnet_v2.preprocess_input(
        image * 255.0
    )
    return tf.expand_dims(model_input, axis=0), display_image.numpy()


def get_backbone_and_head(model):
    backbone = model.get_layer(BACKBONE_LAYER_NAME)
    backbone_index = model.layers.index(backbone)
    head_layers = model.layers[backbone_index + 1:]
    if not head_layers:
        raise ValueError("No classifier head layers found after backbone.")
    return backbone, head_layers


def forward_classifier_head(feature_tensor, head_layers):
    x = feature_tensor
    for layer in head_layers:
        if isinstance(layer, tf.keras.layers.Dropout):
            x = layer(x, training=False)
        else:
            x = layer(x)
    return x


def gradcampp_forward(image_batch, backbone, head_layers):
    """
    Practical Grad-CAM++ implementation using first gradients and
    the standard squared/cubed-gradient alpha weighting.
    """
    with tf.GradientTape() as tape:
        feature_maps = backbone(image_batch, training=False)
        tape.watch(feature_maps)
        prediction = forward_classifier_head(feature_maps, head_layers)
        score = prediction[:, 0]

    gradients = tape.gradient(score, feature_maps)
    if gradients is None:
        raise RuntimeError("Gradient computation returned None.")

    gradients_2 = tf.square(gradients)
    gradients_3 = gradients_2 * gradients

    spatial_activation_sum = tf.reduce_sum(
        feature_maps,
        axis=(1, 2),
        keepdims=True,
    )

    alpha_denominator = (
        2.0 * gradients_2
        + gradients_3 * spatial_activation_sum
    )

    safe_denominator = tf.where(
        tf.abs(alpha_denominator) > EPSILON,
        alpha_denominator,
        tf.ones_like(alpha_denominator),
    )

    alphas = gradients_2 / safe_denominator
    positive_gradients = tf.nn.relu(gradients)

    weights = tf.reduce_sum(
        alphas * positive_gradients,
        axis=(1, 2),
    )

    heatmap = tf.reduce_sum(
        feature_maps
        * weights[:, tf.newaxis, tf.newaxis, :],
        axis=-1,
    )

    heatmap = tf.nn.relu(heatmap)

    heatmap_max = tf.reduce_max(
        heatmap,
        axis=(1, 2),
        keepdims=True,
    )

    heatmap = tf.where(
        heatmap_max > EPSILON,
        heatmap / heatmap_max,
        tf.zeros_like(heatmap),
    )

    return prediction, heatmap


def resize_heatmap_to_image(heatmap):
    heatmap_tensor = tf.convert_to_tensor(
        heatmap[np.newaxis, ..., np.newaxis],
        dtype=tf.float32,
    )
    resized = tf.image.resize(
        heatmap_tensor,
        IMAGE_SIZE,
        method="bilinear",
    )[0, :, :, 0].numpy()

    resized = np.clip(resized, 0.0, None)
    maximum = float(resized.max())
    if maximum > 0:
        resized = resized / maximum
    return resized.astype(np.float32)


def save_heatmap_png(heatmap, output_path):
    image = Image.fromarray(
        np.uint8(np.clip(heatmap, 0.0, 1.0) * 255.0),
        mode="L",
    )
    image.save(output_path)


def save_overlay(display_image, heatmap, output_path):
    grayscale = np.mean(display_image, axis=-1)
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(grayscale, cmap="gray", vmin=0, vmax=1)
    plt.imshow(
        heatmap,
        cmap="viridis",
        alpha=OVERLAY_ALPHA,
        vmin=0,
        vmax=1,
    )
    plt.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)


def choose_montage_examples(result_df):
    selected = []

    if EXISTING_GRADCAM_EXAMPLES_FILE.exists():
        old = pd.read_csv(EXISTING_GRADCAM_EXAMPLES_FILE)

        if "full_image_id" in old.columns:
            ordered_ids = (
                old["full_image_id"]
                .dropna()
                .astype(str)
                .tolist()
            )

            for full_image_id in ordered_ids:
                match = result_df[
                    result_df["full_image_id"].astype(str) == full_image_id
                ]
                if len(match):
                    selected.append(match.iloc[0])

            if len(selected) >= 20:
                return pd.DataFrame(selected[:20])

    selected_frames = []

    for outcome in ["TP", "TN", "FP", "FN"]:
        subset = (
            result_df[result_df["outcome"] == outcome]
            .sort_values(
                "classification_confidence",
                ascending=False,
            )
            .head(5)
        )
        selected_frames.append(subset)

    return pd.concat(selected_frames, ignore_index=True)


def save_montage(example_df):
    if len(example_df) == 0:
        return

    n_columns = 5
    n_rows = int(np.ceil(len(example_df) / n_columns))
    fig = plt.figure(figsize=(4.0 * n_columns, 4.2 * n_rows))

    for index, row in enumerate(
        example_df.itertuples(index=False),
        start=1,
    ):
        ax = fig.add_subplot(n_rows, n_columns, index)

        image_path = resolve_path(
            getattr(row, "preprocessed_path")
        )
        _, display_image = load_model_input_image(image_path)

        heatmap = np.load(
            getattr(row, "heatmap_npy_path")
        )

        grayscale = np.mean(display_image, axis=-1)

        ax.imshow(grayscale, cmap="gray", vmin=0, vmax=1)
        ax.imshow(
            heatmap,
            cmap="viridis",
            alpha=OVERLAY_ALPHA,
            vmin=0,
            vmax=1,
        )
        ax.axis("off")

        ax.set_title(
            f"{getattr(row, 'outcome')} | "
            f"{getattr(row, 'patient_id')}\n"
            f"P(malignant)="
            f"{getattr(row, 'predicted_probability'):.3f}",
            fontsize=9,
        )

    fig.suptitle(
        "Fine-Tuned ResNet50V2 Grad-CAM++ Examples",
        fontsize=16,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.97))

    fig.savefig(
        COMPARISON_FIGURE,
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


for path in [
    MODEL_FILE,
    TEST_PREDICTIONS_FILE,
    ROI_XAI_MANIFEST_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path.resolve()}"
        )


predictions_df = pd.read_csv(TEST_PREDICTIONS_FILE)
roi_df = pd.read_csv(ROI_XAI_MANIFEST_FILE)

prediction_id_column = find_column(
    predictions_df,
    ["full_image_id", "image_id"],
)

prediction_probability_column = find_column(
    predictions_df,
    ["predicted_probability", "probability", "prediction"],
)

prediction_image_path_column = find_column(
    predictions_df,
    ["preprocessed_path", "full_preprocessed_path", "image_path"],
)

prediction_pathology_column = find_column(
    predictions_df,
    ["binary_pathology", "pathology"],
)

prediction_patient_column = find_column(
    predictions_df,
    ["patient_id"],
)

roi_status_column = find_column(
    roi_df,
    ["roi_xai_status"],
)

roi_split_column = find_column(
    roi_df,
    ["full_split", "split"],
)

roi_id_column = find_column(
    roi_df,
    ["full_image_id", "image_id"],
)

successful_test_roi = roi_df[
    (roi_df[roi_status_column] == "success")
    & (roi_df[roi_split_column] == "test")
].copy()

roi_test_ids = set(
    successful_test_roi[roi_id_column].astype(str)
)

cohort = predictions_df[
    predictions_df[prediction_id_column]
    .astype(str)
    .isin(roi_test_ids)
].copy()

cohort = cohort.rename(
    columns={
        prediction_id_column: "full_image_id",
        prediction_probability_column: "predicted_probability",
        prediction_image_path_column: "preprocessed_path",
        prediction_pathology_column: "binary_pathology",
        prediction_patient_column: "patient_id",
    }
)

cohort["full_image_id"] = cohort["full_image_id"].astype(str)

cohort = cohort.drop_duplicates(
    subset=["full_image_id"]
).reset_index(drop=True)

if len(cohort) != 414:
    raise ValueError(
        "Expected exactly 414 ROI-evaluable test mammograms, "
        f"but found {len(cohort)}."
    )

cohort["label"] = cohort[
    "binary_pathology"
].apply(pathology_to_int)

cohort["outcome"] = [
    classify_outcome(
        label,
        probability,
        CLASSIFICATION_THRESHOLD,
    )
    for label, probability in zip(
        cohort["label"],
        cohort["predicted_probability"],
    )
]

cohort["classification_confidence"] = [
    classification_confidence(
        probability,
        outcome,
    )
    for probability, outcome in zip(
        cohort["predicted_probability"],
        cohort["outcome"],
    )
]

if MAX_CASES is not None:
    cohort = cohort.head(int(MAX_CASES)).copy()

cohort["resolved_preprocessed_path"] = cohort[
    "preprocessed_path"
].apply(resolve_path)

cohort["image_exists"] = cohort[
    "resolved_preprocessed_path"
].apply(
    lambda p: p is not None and p.exists()
)

if not cohort["image_exists"].all():
    missing = cohort[~cohort["image_exists"]]
    raise FileNotFoundError(
        "Some preprocessed mammograms are missing:\n"
        + missing[
            [
                "full_image_id",
                "patient_id",
                "preprocessed_path",
            ]
        ].head(20).to_string(index=False)
    )


print("=" * 72)
print("GRAD-CAM++ GENERATION FOR FINE-TUNED RESNET50V2")
print("=" * 72)

print("\nLoading model:")
print(MODEL_FILE.resolve())

model = tf.keras.models.load_model(MODEL_FILE)

backbone, head_layers = get_backbone_and_head(model)

print("\nBackbone:")
print(backbone.name)

print("\nBackbone output shape:")
print(backbone.output_shape)

print("\nClassifier head:")
for layer in head_layers:
    print(
        f"  {layer.name:<25} "
        f"{layer.__class__.__name__}"
    )


sample_row = cohort.iloc[0]
sample_batch, _ = load_model_input_image(
    sample_row["resolved_preprocessed_path"]
)

full_prediction = model(
    sample_batch,
    training=False,
).numpy()[0, 0]

sample_features = backbone(
    sample_batch,
    training=False,
)

manual_prediction = forward_classifier_head(
    sample_features,
    head_layers,
).numpy()[0, 0]

difference = abs(
    float(full_prediction)
    - float(manual_prediction)
)

print("\nForward-pass sanity check:")
print(f"Full model prediction:  {full_prediction:.8f}")
print(f"Manual head prediction: {manual_prediction:.8f}")
print(f"Absolute difference:     {difference:.10f}")

if difference > 1e-5:
    raise RuntimeError(
        "Manual backbone/head forward pass does not match full model."
    )


for directory in [
    OUTPUT_ROOT,
    HEATMAP_NPY_DIR,
    HEATMAP_PNG_DIR,
    OVERLAY_DIR,
    FIGURE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


print("\nROI-evaluable test mammograms:")
print(len(cohort))

print("\nOutcome counts at threshold 0.5:")
print(cohort["outcome"].value_counts())

print("\nGenerating Grad-CAM++ maps...")

results = []
prediction_differences = []
start_time = time.time()

for row in tqdm(
    cohort.itertuples(index=False),
    total=len(cohort),
    desc="Grad-CAM++",
):
    full_image_id = str(
        getattr(row, "full_image_id")
    )

    image_path = Path(
        getattr(
            row,
            "resolved_preprocessed_path",
        )
    )

    image_batch, display_image = load_model_input_image(
        image_path
    )

    prediction, low_resolution_heatmap = gradcampp_forward(
        image_batch,
        backbone,
        head_layers,
    )

    generated_probability = float(
        prediction.numpy()[0, 0]
    )

    saved_probability = float(
        getattr(row, "predicted_probability")
    )

    prediction_difference = abs(
        generated_probability - saved_probability
    )

    prediction_differences.append(
        prediction_difference
    )

    low_resolution_heatmap = (
        low_resolution_heatmap.numpy()[0]
    )

    heatmap = resize_heatmap_to_image(
        low_resolution_heatmap
    )

    npy_path = HEATMAP_NPY_DIR / f"{full_image_id}.npy"
    png_path = HEATMAP_PNG_DIR / f"{full_image_id}.png"
    overlay_path = OVERLAY_DIR / f"{full_image_id}.png"

    np.save(npy_path, heatmap)
    save_heatmap_png(heatmap, png_path)
    save_overlay(
        display_image,
        heatmap,
        overlay_path,
    )

    results.append({
        "full_image_id": full_image_id,
        "patient_id": getattr(row, "patient_id"),
        "binary_pathology": getattr(
            row,
            "binary_pathology",
        ),
        "label": int(getattr(row, "label")),
        "predicted_probability": saved_probability,
        "gradcampp_generated_probability": generated_probability,
        "prediction_absolute_difference": prediction_difference,
        "outcome": getattr(row, "outcome"),
        "classification_confidence": float(
            getattr(
                row,
                "classification_confidence",
            )
        ),
        "preprocessed_path": str(image_path),
        "heatmap_npy_path": str(npy_path.resolve()),
        "heatmap_png_path": str(png_path.resolve()),
        "overlay_path": str(overlay_path.resolve()),
        "heatmap_min": float(heatmap.min()),
        "heatmap_max": float(heatmap.max()),
        "heatmap_mean": float(heatmap.mean()),
        "heatmap_nonzero_fraction": float(
            np.mean(heatmap > 0)
        ),
    })


result_df = pd.DataFrame(results)
result_df.to_csv(
    OUTPUT_MANIFEST,
    index=False,
)

example_df = choose_montage_examples(result_df)
save_montage(example_df)

elapsed_seconds = time.time() - start_time

maximum_prediction_difference = (
    float(max(prediction_differences))
    if prediction_differences
    else 0.0
)

summary = {
    "method": "Grad-CAM++",
    "model": str(MODEL_FILE),
    "backbone": BACKBONE_LAYER_NAME,
    "target": "malignancy sigmoid probability",
    "classification_threshold": float(
        CLASSIFICATION_THRESHOLD
    ),
    "mammograms_processed": int(len(result_df)),
    "outcome_counts": {
        str(key): int(value)
        for key, value in result_df[
            "outcome"
        ].value_counts().items()
    },
    "maximum_prediction_difference": maximum_prediction_difference,
    "heatmap_size": [
        int(IMAGE_SIZE[0]),
        int(IMAGE_SIZE[1]),
    ],
    "raw_backbone_heatmap_size": [
        int(backbone.output_shape[1]),
        int(backbone.output_shape[2]),
    ],
    "elapsed_seconds": float(elapsed_seconds),
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


print("\n")
print("=" * 72)
print("GRAD-CAM++ GENERATION COMPLETE")
print("=" * 72)

print(
    f"\nGrad-CAM++ maps generated: "
    f"{len(result_df):,}"
)

print("\nExamples by category:")
print(result_df["outcome"].value_counts())

print("\nHeatmap dimensions:")
print(f"{IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}")

print("\nMaximum prediction difference:")
print(f"{maximum_prediction_difference:.10f}")

print(
    f"\nElapsed time: "
    f"{elapsed_seconds / 60.0:.1f} minutes"
)

print("\nSaved manifest:")
print(OUTPUT_MANIFEST.resolve())

print("\nSaved raw Grad-CAM++ arrays:")
print(HEATMAP_NPY_DIR.resolve())

print("\nSaved heatmap PNGs:")
print(HEATMAP_PNG_DIR.resolve())

print("\nSaved overlays:")
print(OVERLAY_DIR.resolve())

print("\nSaved comparison figure:")
print(COMPARISON_FIGURE.resolve())

print("\nSaved summary:")
print(SUMMARY_FILE.resolve())
