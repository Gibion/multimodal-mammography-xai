"""
Generate Integrated Gradients explanation maps for the fine-tuned
ResNet50V2 classifier on the exact 414-image ROI-evaluable test cohort.
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
TEST_PREDICTIONS_FILE = Path("results/resnet50v2_finetuned/evaluation/test_predictions.csv")
ROI_XAI_MANIFEST_FILE = Path("data/processed/cbis_ddsm/manifests/cbis_ddsm_roi_xai_manifest.csv")
EXISTING_GRADCAM_EXAMPLES_FILE = Path("results/resnet50v2_finetuned/gradcam/gradcam_examples_manifest.csv")

OUTPUT_ROOT = Path("results/resnet50v2_finetuned/integrated_gradients")
ATTRIBUTION_NPY_DIR = OUTPUT_ROOT / "attributions_npy"
HEATMAP_PNG_DIR = OUTPUT_ROOT / "heatmaps_png"
OVERLAY_DIR = OUTPUT_ROOT / "overlays"
FIGURE_DIR = OUTPUT_ROOT / "figures"
OUTPUT_MANIFEST = OUTPUT_ROOT / "integrated_gradients_test_manifest.csv"
SUMMARY_FILE = OUTPUT_ROOT / "integrated_gradients_summary.json"
COMPARISON_FIGURE = FIGURE_DIR / "integrated_gradients_tp_tn_fp_fn_examples.png"

IMAGE_SIZE = (512, 512)
CLASSIFICATION_THRESHOLD = 0.5
N_STEPS = 200
IG_BATCH_SIZE = 10
BASELINE_VALUE = -1.0  # black RGB after ResNet50V2 preprocessing
OVERLAY_ALPHA = 0.45
EPSILON = 1e-10
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
        raise ValueError("Could not find any of these columns: " + ", ".join(candidates))
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
    return float(probability if outcome in {"TP", "FP"} else 1.0 - probability)


def load_model_input_image(path):
    image_bytes = tf.io.read_file(str(path))
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")
    display_image = tf.clip_by_value(image, 0.0, 1.0)
    model_input = tf.keras.applications.resnet_v2.preprocess_input(image * 255.0)
    return tf.expand_dims(model_input, axis=0), display_image.numpy()


def make_baseline_like(image_batch):
    return tf.ones_like(image_batch, dtype=tf.float32) * BASELINE_VALUE


def interpolate_inputs(baseline, image, alphas):
    alphas = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    return baseline[0][tf.newaxis, ...] + alphas * (
        image[0][tf.newaxis, ...] - baseline[0][tf.newaxis, ...]
    )


def input_gradients(model, interpolated_batch):
    with tf.GradientTape() as tape:
        tape.watch(interpolated_batch)
        scores = model(interpolated_batch, training=False)[:, 0]
    gradients = tape.gradient(scores, interpolated_batch)
    if gradients is None:
        raise RuntimeError("Integrated Gradients input gradient returned None.")
    return gradients


def compute_integrated_gradients(model, image_batch, n_steps=N_STEPS, integration_batch_size=IG_BATCH_SIZE):
    baseline = make_baseline_like(image_batch)
    baseline_probability = float(model(baseline, training=False).numpy()[0, 0])
    image_probability = float(model(image_batch, training=False).numpy()[0, 0])

    alphas = tf.linspace(0.0, 1.0, n_steps + 1)
    gradient_batches = []

    for start in range(0, n_steps + 1, integration_batch_size):
        stop = min(start + integration_batch_size, n_steps + 1)
        interpolated_batch = interpolate_inputs(baseline, image_batch, alphas[start:stop])
        gradient_batches.append(input_gradients(model, interpolated_batch))

    gradients = tf.concat(gradient_batches, axis=0)
    trapezoids = (gradients[:-1] + gradients[1:]) / 2.0
    average_gradients = tf.reduce_mean(trapezoids, axis=0)
    attributions = (image_batch[0] - baseline[0]) * average_gradients

    attribution_sum = float(tf.reduce_sum(attributions).numpy())
    prediction_difference = image_probability - baseline_probability
    completeness_delta = attribution_sum - prediction_difference

    return (
        attributions.numpy().astype(np.float32),
        baseline_probability,
        image_probability,
        attribution_sum,
        float(completeness_delta),
    )


def collapse_attributions_to_heatmap(attributions):
    # Positive evidence for the malignant-class score.
    positive = np.maximum(attributions, 0.0)
    heatmap = np.sum(positive, axis=-1)
    heatmap = np.clip(heatmap, 0.0, None)
    maximum = float(heatmap.max())
    if maximum > EPSILON:
        heatmap = heatmap / maximum
    else:
        heatmap = np.zeros_like(heatmap, dtype=np.float32)
    return heatmap.astype(np.float32)


def save_heatmap_png(heatmap, output_path):
    Image.fromarray(
        np.uint8(np.clip(heatmap, 0.0, 1.0) * 255.0),
        mode="L",
    ).save(output_path)


def save_overlay(display_image, heatmap, output_path):
    grayscale = np.mean(display_image, axis=-1)
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(grayscale, cmap="gray", vmin=0, vmax=1)
    plt.imshow(heatmap, cmap="viridis", alpha=OVERLAY_ALPHA, vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def choose_montage_examples(result_df):
    selected = []
    if EXISTING_GRADCAM_EXAMPLES_FILE.exists():
        old = pd.read_csv(EXISTING_GRADCAM_EXAMPLES_FILE)
        if "full_image_id" in old.columns:
            for full_image_id in old["full_image_id"].dropna().astype(str).tolist():
                match = result_df[result_df["full_image_id"].astype(str) == full_image_id]
                if len(match):
                    selected.append(match.iloc[0])
            if len(selected) >= 20:
                return pd.DataFrame(selected[:20])

    frames = []
    for outcome in ["TP", "TN", "FP", "FN"]:
        frames.append(
            result_df[result_df["outcome"] == outcome]
            .sort_values("classification_confidence", ascending=False)
            .head(5)
        )
    return pd.concat(frames, ignore_index=True)


def save_montage(example_df):
    if len(example_df) == 0:
        return
    n_columns = 5
    n_rows = int(np.ceil(len(example_df) / n_columns))
    fig = plt.figure(figsize=(4.0 * n_columns, 4.2 * n_rows))

    for index, row in enumerate(example_df.itertuples(index=False), start=1):
        ax = fig.add_subplot(n_rows, n_columns, index)
        _, display_image = load_model_input_image(resolve_path(getattr(row, "preprocessed_path")))
        heatmap = np.load(getattr(row, "heatmap_npy_path"))
        grayscale = np.mean(display_image, axis=-1)
        ax.imshow(grayscale, cmap="gray", vmin=0, vmax=1)
        ax.imshow(heatmap, cmap="viridis", alpha=OVERLAY_ALPHA, vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(
            f"{getattr(row, 'outcome')} | {getattr(row, 'patient_id')}\n"
            f"P(malignant)={getattr(row, 'predicted_probability'):.3f}",
            fontsize=9,
        )

    fig.suptitle("Fine-Tuned ResNet50V2 Integrated Gradients Examples", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(COMPARISON_FIGURE, dpi=250, bbox_inches="tight")
    plt.close(fig)


for path in [MODEL_FILE, TEST_PREDICTIONS_FILE, ROI_XAI_MANIFEST_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path.resolve()}")

predictions_df = pd.read_csv(TEST_PREDICTIONS_FILE)
roi_df = pd.read_csv(ROI_XAI_MANIFEST_FILE)

prediction_id_column = find_column(predictions_df, ["full_image_id", "image_id"])
prediction_probability_column = find_column(predictions_df, ["predicted_probability", "probability", "prediction"])
prediction_image_path_column = find_column(predictions_df, ["preprocessed_path", "full_preprocessed_path", "image_path"])
prediction_pathology_column = find_column(predictions_df, ["binary_pathology", "pathology"])
prediction_patient_column = find_column(predictions_df, ["patient_id"])
roi_status_column = find_column(roi_df, ["roi_xai_status"])
roi_split_column = find_column(roi_df, ["full_split", "split"])
roi_id_column = find_column(roi_df, ["full_image_id", "image_id"])

successful_test_roi = roi_df[
    (roi_df[roi_status_column] == "success")
    & (roi_df[roi_split_column] == "test")
].copy()
roi_test_ids = set(successful_test_roi[roi_id_column].astype(str))

cohort = predictions_df[
    predictions_df[prediction_id_column].astype(str).isin(roi_test_ids)
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
cohort = cohort.drop_duplicates(subset=["full_image_id"]).reset_index(drop=True)

if len(cohort) != 414:
    raise ValueError(
        "Expected exactly 414 ROI-evaluable test mammograms, "
        f"but found {len(cohort)}."
    )

cohort["label"] = cohort["binary_pathology"].apply(pathology_to_int)
cohort["outcome"] = [
    classify_outcome(label, probability, CLASSIFICATION_THRESHOLD)
    for label, probability in zip(cohort["label"], cohort["predicted_probability"])
]
cohort["classification_confidence"] = [
    classification_confidence(probability, outcome)
    for probability, outcome in zip(cohort["predicted_probability"], cohort["outcome"])
]

if MAX_CASES is not None:
    cohort = cohort.head(int(MAX_CASES)).copy()

cohort["resolved_preprocessed_path"] = cohort["preprocessed_path"].apply(resolve_path)
cohort["image_exists"] = cohort["resolved_preprocessed_path"].apply(
    lambda path: path is not None and path.exists()
)

if not cohort["image_exists"].all():
    missing = cohort[~cohort["image_exists"]]
    raise FileNotFoundError(
        "Some preprocessed mammograms are missing:\n"
        + missing[["full_image_id", "patient_id", "preprocessed_path"]]
        .head(20)
        .to_string(index=False)
    )

print("=" * 72)
print("INTEGRATED GRADIENTS FOR FINE-TUNED RESNET50V2")
print("=" * 72)
print("\nLoading model:")
print(MODEL_FILE.resolve())

model = tf.keras.models.load_model(MODEL_FILE)

print("\nModel input shape:")
print(model.input_shape)
print("\nIntegration settings:")
print(f"Steps: {N_STEPS}")
print(f"Interpolation batch size: {IG_BATCH_SIZE}")
print(f"Baseline value in model-input space: {BASELINE_VALUE}")

sample_row = cohort.iloc[0]
sample_batch, _ = load_model_input_image(sample_row["resolved_preprocessed_path"])
sample_prediction = float(model(sample_batch, training=False).numpy()[0, 0])
sample_saved_probability = float(sample_row["predicted_probability"])
sample_difference = abs(sample_prediction - sample_saved_probability)
sample_baseline = make_baseline_like(sample_batch)
sample_baseline_prediction = float(model(sample_baseline, training=False).numpy()[0, 0])

print("\nForward-pass sanity check:")
print(f"Saved prediction:         {sample_saved_probability:.8f}")
print(f"Generated prediction:     {sample_prediction:.8f}")
print(f"Absolute difference:      {sample_difference:.10f}")
print(f"Black-baseline prediction:{sample_baseline_prediction:.8f}")

if sample_difference > 1e-5:
    raise RuntimeError("Generated prediction does not match saved CNN prediction.")

for directory in [OUTPUT_ROOT, ATTRIBUTION_NPY_DIR, HEATMAP_PNG_DIR, OVERLAY_DIR, FIGURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print("\nROI-evaluable test mammograms:")
print(len(cohort))
print("\nOutcome counts at threshold 0.5:")
print(cohort["outcome"].value_counts())
print("\nGenerating Integrated Gradients maps...")

results = []
prediction_differences = []
absolute_completeness_deltas = []
relative_completeness_errors = []
start_time = time.time()

for row in tqdm(cohort.itertuples(index=False), total=len(cohort), desc="Integrated Gradients"):
    full_image_id = str(getattr(row, "full_image_id"))
    image_path = Path(getattr(row, "resolved_preprocessed_path"))
    image_batch, display_image = load_model_input_image(image_path)

    (
        attributions,
        baseline_probability,
        generated_probability,
        attribution_sum,
        completeness_delta,
    ) = compute_integrated_gradients(
        model,
        image_batch,
        n_steps=N_STEPS,
        integration_batch_size=IG_BATCH_SIZE,
    )

    saved_probability = float(getattr(row, "predicted_probability"))
    prediction_difference = abs(generated_probability - saved_probability)
    prediction_differences.append(prediction_difference)

    prediction_minus_baseline = generated_probability - baseline_probability
    absolute_completeness_delta = abs(completeness_delta)
    relative_completeness_error = absolute_completeness_delta / max(
        abs(prediction_minus_baseline), EPSILON
    )
    absolute_completeness_deltas.append(absolute_completeness_delta)
    relative_completeness_errors.append(relative_completeness_error)

    heatmap = collapse_attributions_to_heatmap(attributions)

    npy_path = ATTRIBUTION_NPY_DIR / f"{full_image_id}.npy"
    png_path = HEATMAP_PNG_DIR / f"{full_image_id}.png"
    overlay_path = OVERLAY_DIR / f"{full_image_id}.png"

    np.save(npy_path, heatmap)
    save_heatmap_png(heatmap, png_path)
    save_overlay(display_image, heatmap, overlay_path)

    results.append({
        "full_image_id": full_image_id,
        "patient_id": getattr(row, "patient_id"),
        "binary_pathology": getattr(row, "binary_pathology"),
        "label": int(getattr(row, "label")),
        "predicted_probability": saved_probability,
        "ig_generated_probability": generated_probability,
        "prediction_absolute_difference": prediction_difference,
        "baseline_probability": baseline_probability,
        "prediction_minus_baseline": prediction_minus_baseline,
        "ig_attribution_sum_signed": attribution_sum,
        "ig_completeness_delta_signed": completeness_delta,
        "ig_completeness_absolute_error": absolute_completeness_delta,
        "ig_completeness_relative_error": relative_completeness_error,
        "integration_steps": int(N_STEPS),
        "outcome": getattr(row, "outcome"),
        "classification_confidence": float(getattr(row, "classification_confidence")),
        "preprocessed_path": str(image_path),
        "heatmap_npy_path": str(npy_path.resolve()),
        "heatmap_png_path": str(png_path.resolve()),
        "overlay_path": str(overlay_path.resolve()),
        "heatmap_min": float(heatmap.min()),
        "heatmap_max": float(heatmap.max()),
        "heatmap_mean": float(heatmap.mean()),
        "heatmap_nonzero_fraction": float(np.mean(heatmap > 0)),
    })

result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_MANIFEST, index=False)

example_df = choose_montage_examples(result_df)
save_montage(example_df)

elapsed_seconds = time.time() - start_time
maximum_prediction_difference = float(max(prediction_differences)) if prediction_differences else 0.0
mean_absolute_completeness_error = float(np.mean(absolute_completeness_deltas)) if absolute_completeness_deltas else 0.0
median_absolute_completeness_error = float(np.median(absolute_completeness_deltas)) if absolute_completeness_deltas else 0.0
mean_relative_completeness_error = float(np.mean(relative_completeness_errors)) if relative_completeness_errors else 0.0
median_relative_completeness_error = float(np.median(relative_completeness_errors)) if relative_completeness_errors else 0.0

summary = {
    "method": "Integrated Gradients",
    "model": str(MODEL_FILE),
    "target": "malignancy sigmoid probability",
    "classification_threshold": float(CLASSIFICATION_THRESHOLD),
    "baseline": {
        "description": "black image in original RGB space",
        "model_input_value": float(BASELINE_VALUE),
    },
    "integration_steps": int(N_STEPS),
    "integration_batch_size": int(IG_BATCH_SIZE),
    "attribution_channel_collapse": "sum of positive channel attributions",
    "mammograms_processed": int(len(result_df)),
    "outcome_counts": {
        str(key): int(value)
        for key, value in result_df["outcome"].value_counts().items()
    },
    "maximum_prediction_difference": maximum_prediction_difference,
    "completeness": {
        "mean_absolute_error": mean_absolute_completeness_error,
        "median_absolute_error": median_absolute_completeness_error,
        "mean_relative_error": mean_relative_completeness_error,
        "median_relative_error": median_relative_completeness_error,
    },
    "heatmap_size": [int(IMAGE_SIZE[0]), int(IMAGE_SIZE[1])],
    "elapsed_seconds": float(elapsed_seconds),
}

with SUMMARY_FILE.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print("\n" + "=" * 72)
print("INTEGRATED GRADIENTS GENERATION COMPLETE")
print("=" * 72)
print(f"\nIntegrated Gradients maps generated: {len(result_df):,}")
print("\nExamples by category:")
print(result_df["outcome"].value_counts())
print("\nHeatmap dimensions:")
print(f"{IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}")
print("\nMaximum prediction difference:")
print(f"{maximum_prediction_difference:.10f}")
print("\nCompleteness QA:")
print(f"Mean absolute error:   {mean_absolute_completeness_error:.6f}")
print(f"Median absolute error: {median_absolute_completeness_error:.6f}")
print(f"Mean relative error:   {mean_relative_completeness_error:.6f}")
print(f"Median relative error: {median_relative_completeness_error:.6f}")
print(f"\nElapsed time: {elapsed_seconds / 60.0:.1f} minutes")
print("\nSaved manifest:")
print(OUTPUT_MANIFEST.resolve())
print("\nSaved raw Integrated Gradients maps:")
print(ATTRIBUTION_NPY_DIR.resolve())
print("\nSaved heatmap PNGs:")
print(HEATMAP_PNG_DIR.resolve())
print("\nSaved overlays:")
print(OVERLAY_DIR.resolve())
print("\nSaved comparison figure:")
print(COMPARISON_FIGURE.resolve())
print("\nSaved summary:")
print(SUMMARY_FILE.resolve())
