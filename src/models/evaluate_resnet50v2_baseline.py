from pathlib import Path
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path.cwd()
MANIFEST_FILE = Path(
    "data/processed/cbis_ddsm/manifests/"
    "cbis_ddsm_full_image_manifest_preprocessed_v4.csv"
)
MODEL_PATH = Path("models/resnet50v2_baseline/best_model.keras")
RESULTS_DIR = Path("results/resnet50v2_baseline/evaluation")
FIGURES_DIR = RESULTS_DIR / "figures"

IMAGE_SIZE = 512
BATCH_SIZE = 4
DEFAULT_THRESHOLD = 0.5
AUTOTUNE = tf.data.AUTOTUNE

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST_FILE)

required_columns = [
    "preprocessed_path",
    "preprocessing_status",
    "binary_pathology",
    "split",
    "patient_id",
    "image_id",
]

missing_columns = [c for c in required_columns if c not in df.columns]
if missing_columns:
    raise ValueError(
        "Manifest is missing required columns: "
        + ", ".join(missing_columns)
    )

df = df[df["preprocessing_status"] == "success"].copy()


def resolve_path(value):
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


df["resolved_path"] = df["preprocessed_path"].apply(resolve_path)

exists = df["resolved_path"].apply(os.path.exists)

print("=" * 72)
print("RESNET50V2 BASELINE EVALUATION")
print("=" * 72)
print(f"\nManifest rows after filtering: {len(df):,}")
print(f"Source images found: {exists.sum():,} / {len(exists):,}")

if not exists.all():
    missing = df.loc[~exists, "resolved_path"]
    print("\nFirst missing images:")
    for path in missing.head(10):
        print(path)
    raise FileNotFoundError(
        f"{(~exists).sum():,} source images could not be found."
    )

label_map = {"BENIGN": 0, "MALIGNANT": 1}

df["label"] = (
    df["binary_pathology"]
    .astype(str)
    .str.upper()
    .map(label_map)
)

if df["label"].isna().any():
    unknown = (
        df.loc[df["label"].isna(), "binary_pathology"]
        .unique()
        .tolist()
    )
    raise ValueError(f"Unexpected pathology labels: {unknown}")

df["label"] = df["label"].astype(np.int32)

validation_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

print("\nValidation set:")
print(validation_df["binary_pathology"].value_counts())

print("\nTest set:")
print(test_df["binary_pathology"].value_counts())


def load_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_bytes, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image * 255.0
    image = tf.ensure_shape(
        image,
        [IMAGE_SIZE, IMAGE_SIZE, 1]
    )
    image = tf.image.grayscale_to_rgb(image)
    image = tf.ensure_shape(
        image,
        [IMAGE_SIZE, IMAGE_SIZE, 3]
    )
    image = tf.keras.applications.resnet_v2.preprocess_input(image)
    return image, label


def build_dataset(frame):
    paths = frame["resolved_path"].to_numpy()
    labels = frame["label"].to_numpy(dtype=np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


validation_dataset = build_dataset(validation_df)
test_dataset = build_dataset(test_df)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH.resolve()}"
    )

print("\nLoading model:")
print(MODEL_PATH.resolve())

model = tf.keras.models.load_model(MODEL_PATH)


def predict_probabilities(dataset, frame, split_name):
    print(f"\nPredicting probabilities for {split_name} set...")
    probabilities = model.predict(dataset, verbose=1).reshape(-1)

    if len(probabilities) != len(frame):
        raise ValueError(
            f"Prediction count mismatch for {split_name}: "
            f"{len(probabilities)} vs {len(frame)}"
        )

    return probabilities


validation_probabilities = predict_probabilities(
    validation_dataset,
    validation_df,
    "validation"
)

test_probabilities = predict_probabilities(
    test_dataset,
    test_df,
    "test"
)

validation_labels = validation_df["label"].to_numpy()
test_labels = test_df["label"].to_numpy()

# Threshold chosen ONLY on validation data.
threshold_candidates = np.linspace(0.0, 1.0, 1001)
validation_f1_scores = []

for threshold in threshold_candidates:
    preds = (validation_probabilities >= threshold).astype(int)
    validation_f1_scores.append(
        f1_score(validation_labels, preds, zero_division=0)
    )

best_threshold_index = int(np.argmax(validation_f1_scores))
best_validation_threshold = float(
    threshold_candidates[best_threshold_index]
)
best_validation_f1 = float(
    validation_f1_scores[best_threshold_index]
)

print("\nThreshold selection:")
print(f"Default threshold: {DEFAULT_THRESHOLD:.3f}")
print(
    "Validation-selected threshold (max F1): "
    f"{best_validation_threshold:.3f}"
)
print(
    "Validation F1 at selected threshold: "
    f"{best_validation_f1:.4f}"
)


def calculate_metrics(labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(
            average_precision_score(labels, probabilities)
        ),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall_sensitivity": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "specificity": float(specificity),
        "f1": float(
            f1_score(labels, predictions, zero_division=0)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


validation_default_metrics = calculate_metrics(
    validation_labels,
    validation_probabilities,
    DEFAULT_THRESHOLD
)

validation_selected_metrics = calculate_metrics(
    validation_labels,
    validation_probabilities,
    best_validation_threshold
)

test_default_metrics = calculate_metrics(
    test_labels,
    test_probabilities,
    DEFAULT_THRESHOLD
)

test_selected_metrics = calculate_metrics(
    test_labels,
    test_probabilities,
    best_validation_threshold
)


def print_metrics(title, metrics):
    print("\n" + title)
    print("-" * len(title))

    for key in [
        "threshold",
        "accuracy",
        "roc_auc",
        "average_precision",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
    ]:
        print(f"{key:20s}: {metrics[key]:.4f}")

    print(
        "confusion matrix: "
        f"TN={metrics['tn']}, "
        f"FP={metrics['fp']}, "
        f"FN={metrics['fn']}, "
        f"TP={metrics['tp']}"
    )


print_metrics(
    "Validation metrics at threshold 0.5",
    validation_default_metrics
)
print_metrics(
    "Validation metrics at selected threshold",
    validation_selected_metrics
)
print_metrics(
    "Test metrics at threshold 0.5",
    test_default_metrics
)
print_metrics(
    "Test metrics at validation-selected threshold",
    test_selected_metrics
)

# ROC curve
validation_fpr, validation_tpr, _ = roc_curve(
    validation_labels,
    validation_probabilities
)
test_fpr, test_tpr, _ = roc_curve(
    test_labels,
    test_probabilities
)

fig = plt.figure(figsize=(7, 6))
plt.plot(
    validation_fpr,
    validation_tpr,
    label=(
        f"Validation AUC = "
        f"{validation_default_metrics['roc_auc']:.3f}"
    )
)
plt.plot(
    test_fpr,
    test_tpr,
    label=(
        f"Test AUC = "
        f"{test_default_metrics['roc_auc']:.3f}"
    )
)
plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ResNet50V2 Baseline ROC Curve")
plt.legend()
plt.tight_layout()
fig.savefig(
    FIGURES_DIR / "roc_curve.png",
    dpi=180,
    bbox_inches="tight"
)
plt.close(fig)

# Precision-recall curve
val_precision_curve, val_recall_curve, _ = precision_recall_curve(
    validation_labels,
    validation_probabilities
)
test_precision_curve, test_recall_curve, _ = precision_recall_curve(
    test_labels,
    test_probabilities
)

fig = plt.figure(figsize=(7, 6))
plt.plot(
    val_recall_curve,
    val_precision_curve,
    label=(
        f"Validation AP = "
        f"{validation_default_metrics['average_precision']:.3f}"
    )
)
plt.plot(
    test_recall_curve,
    test_precision_curve,
    label=(
        f"Test AP = "
        f"{test_default_metrics['average_precision']:.3f}"
    )
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("ResNet50V2 Baseline Precision-Recall Curve")
plt.legend()
plt.tight_layout()
fig.savefig(
    FIGURES_DIR / "precision_recall_curve.png",
    dpi=180,
    bbox_inches="tight"
)
plt.close(fig)


def save_confusion_matrix(metrics, title, filename):
    matrix = np.array([
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]],
    ])

    fig = plt.figure(figsize=(6, 5))
    plt.imshow(matrix)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0, 1], ["Benign", "Malignant"])
    plt.yticks([0, 1], ["Benign", "Malignant"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()
    fig.savefig(
        FIGURES_DIR / filename,
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)


save_confusion_matrix(
    test_default_metrics,
    "Test Confusion Matrix (Threshold = 0.5)",
    "confusion_matrix_test_threshold_0_5.png"
)

save_confusion_matrix(
    test_selected_metrics,
    (
        "Test Confusion Matrix "
        f"(Validation Threshold = "
        f"{best_validation_threshold:.3f})"
    ),
    "confusion_matrix_test_selected_threshold.png"
)

# Probability distributions
benign_probs = test_probabilities[test_labels == 0]
malignant_probs = test_probabilities[test_labels == 1]

fig = plt.figure(figsize=(8, 6))
plt.hist(
    benign_probs,
    bins=25,
    alpha=0.6,
    label="Benign"
)
plt.hist(
    malignant_probs,
    bins=25,
    alpha=0.6,
    label="Malignant"
)
plt.axvline(
    DEFAULT_THRESHOLD,
    linestyle="--",
    label="Threshold 0.5"
)
plt.axvline(
    best_validation_threshold,
    linestyle=":",
    label=(
        f"Validation-selected threshold "
        f"{best_validation_threshold:.3f}"
    )
)
plt.xlabel("Predicted probability of malignancy")
plt.ylabel("Number of mammograms")
plt.title("Test Predicted Probability Distribution")
plt.legend()
plt.tight_layout()
fig.savefig(
    FIGURES_DIR / "test_probability_distribution.png",
    dpi=180,
    bbox_inches="tight"
)
plt.close(fig)

# Validation threshold selection figure
fig = plt.figure(figsize=(8, 5))
plt.plot(
    threshold_candidates,
    validation_f1_scores
)
plt.axvline(
    best_validation_threshold,
    linestyle="--",
    label=f"Selected = {best_validation_threshold:.3f}"
)
plt.xlabel("Classification Threshold")
plt.ylabel("Validation F1 Score")
plt.title("Validation Threshold Selection")
plt.legend()
plt.tight_layout()
fig.savefig(
    FIGURES_DIR / "validation_threshold_selection.png",
    dpi=180,
    bbox_inches="tight"
)
plt.close(fig)

# Save per-image predictions
validation_predictions = validation_df[
    [
        "image_id",
        "patient_id",
        "binary_pathology",
        "label",
        "preprocessed_path",
    ]
].copy()

validation_predictions["predicted_probability"] = (
    validation_probabilities
)
validation_predictions["prediction_threshold_0_5"] = (
    validation_probabilities >= DEFAULT_THRESHOLD
).astype(int)
validation_predictions["prediction_selected_threshold"] = (
    validation_probabilities >= best_validation_threshold
).astype(int)

test_predictions = test_df[
    [
        "image_id",
        "patient_id",
        "binary_pathology",
        "label",
        "preprocessed_path",
    ]
].copy()

test_predictions["predicted_probability"] = test_probabilities
test_predictions["prediction_threshold_0_5"] = (
    test_probabilities >= DEFAULT_THRESHOLD
).astype(int)
test_predictions["prediction_selected_threshold"] = (
    test_probabilities >= best_validation_threshold
).astype(int)

validation_predictions.to_csv(
    RESULTS_DIR / "validation_predictions.csv",
    index=False
)
test_predictions.to_csv(
    RESULTS_DIR / "test_predictions.csv",
    index=False
)

summary = {
    "model": str(MODEL_PATH),
    "threshold_selection": {
        "method": "max_validation_f1",
        "default_threshold": float(DEFAULT_THRESHOLD),
        "selected_threshold": float(best_validation_threshold),
        "best_validation_f1": float(best_validation_f1),
    },
    "validation": {
        "threshold_0_5": validation_default_metrics,
        "selected_threshold": validation_selected_metrics,
    },
    "test": {
        "threshold_0_5": test_default_metrics,
        "selected_threshold": test_selected_metrics,
    },
}

with (
    RESULTS_DIR / "evaluation_metrics.json"
).open(
    "w",
    encoding="utf-8"
) as handle:
    json.dump(summary, handle, indent=2)

metrics_table = pd.DataFrame([
    {
        "split": "validation",
        "threshold_type": "0.5",
        **validation_default_metrics,
    },
    {
        "split": "validation",
        "threshold_type": "validation_selected",
        **validation_selected_metrics,
    },
    {
        "split": "test",
        "threshold_type": "0.5",
        **test_default_metrics,
    },
    {
        "split": "test",
        "threshold_type": "validation_selected",
        **test_selected_metrics,
    },
])

metrics_table.to_csv(
    RESULTS_DIR / "evaluation_metrics.csv",
    index=False
)

print("\n")
print("=" * 72)
print("BASELINE EVALUATION COMPLETE")
print("=" * 72)

print(
    f"\nValidation-selected threshold: "
    f"{best_validation_threshold:.3f}"
)
print(
    f"Test ROC AUC: "
    f"{test_default_metrics['roc_auc']:.4f}"
)
print(
    f"Test average precision: "
    f"{test_default_metrics['average_precision']:.4f}"
)

print("\nTest threshold 0.5:")
print(
    f"  accuracy    = "
    f"{test_default_metrics['accuracy']:.4f}"
)
print(
    f"  precision   = "
    f"{test_default_metrics['precision']:.4f}"
)
print(
    f"  recall      = "
    f"{test_default_metrics['recall_sensitivity']:.4f}"
)
print(
    f"  specificity = "
    f"{test_default_metrics['specificity']:.4f}"
)
print(
    f"  F1          = "
    f"{test_default_metrics['f1']:.4f}"
)

print("\nTest validation-selected threshold:")
print(
    f"  threshold   = "
    f"{best_validation_threshold:.3f}"
)
print(
    f"  accuracy    = "
    f"{test_selected_metrics['accuracy']:.4f}"
)
print(
    f"  precision   = "
    f"{test_selected_metrics['precision']:.4f}"
)
print(
    f"  recall      = "
    f"{test_selected_metrics['recall_sensitivity']:.4f}"
)
print(
    f"  specificity = "
    f"{test_selected_metrics['specificity']:.4f}"
)
print(
    f"  F1          = "
    f"{test_selected_metrics['f1']:.4f}"
)

print("\nSaved evaluation results:")
print(RESULTS_DIR.resolve())
