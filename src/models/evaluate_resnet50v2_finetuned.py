from pathlib import Path
import json, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve
)

PROJECT_ROOT = Path.cwd()
MANIFEST_FILE = Path("data/processed/cbis_ddsm/manifests/cbis_ddsm_full_image_manifest_preprocessed_v4.csv")
MODEL_PATH = Path("models/resnet50v2_finetuned/best_model.keras")
RESULTS_DIR = Path("results/resnet50v2_finetuned/evaluation")
FIGURES_DIR = RESULTS_DIR / "figures"
IMAGE_SIZE = 512
BATCH_SIZE = 4
DEFAULT_THRESHOLD = 0.5
AUTOTUNE = tf.data.AUTOTUNE

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST_FILE)
df = df[df["preprocessing_status"] == "success"].copy()

def resolve_path(value):
    p = Path(str(value))
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p.resolve())

df["resolved_path"] = df["preprocessed_path"].apply(resolve_path)
exists = df["resolved_path"].apply(os.path.exists)

print("=" * 72)
print("RESNET50V2 FINE-TUNED MODEL EVALUATION")
print("=" * 72)
print(f"\nManifest rows after filtering: {len(df):,}")
print(f"Source images found: {exists.sum():,} / {len(exists):,}")

if not exists.all():
    raise FileNotFoundError(f"{(~exists).sum():,} source images could not be found.")

label_map = {"BENIGN": 0, "MALIGNANT": 1}
df["label"] = df["binary_pathology"].astype(str).str.upper().map(label_map)
if df["label"].isna().any():
    raise ValueError("Unexpected pathology label found.")
df["label"] = df["label"].astype(np.int32)

validation_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

print("\nValidation set:")
print(validation_df["binary_pathology"].value_counts())
print("\nTest set:")
print(test_df["binary_pathology"].value_counts())

def load_image(path, label):
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32) * 255.0
    image = tf.ensure_shape(image, [IMAGE_SIZE, IMAGE_SIZE, 1])
    image = tf.image.grayscale_to_rgb(image)
    image = tf.ensure_shape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.keras.applications.resnet_v2.preprocess_input(image)
    return image, label

def build_dataset(frame):
    ds = tf.data.Dataset.from_tensor_slices((
        frame["resolved_path"].to_numpy(),
        frame["label"].to_numpy(dtype=np.float32)
    ))
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

validation_dataset = build_dataset(validation_df)
test_dataset = build_dataset(test_df)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH.resolve()}")

print("\nLoading fine-tuned model:")
print(MODEL_PATH.resolve())
model = tf.keras.models.load_model(MODEL_PATH)

def predict_probabilities(dataset, frame, split_name):
    print(f"\nPredicting probabilities for {split_name} set...")
    probs = model.predict(dataset, verbose=1).reshape(-1)
    if len(probs) != len(frame):
        raise ValueError("Prediction count mismatch.")
    return probs

val_probs = predict_probabilities(validation_dataset, validation_df, "validation")
test_probs = predict_probabilities(test_dataset, test_df, "test")
val_labels = validation_df["label"].to_numpy()
test_labels = test_df["label"].to_numpy()

thresholds = np.linspace(0.0, 1.0, 1001)
val_f1s = [
    f1_score(val_labels, (val_probs >= t).astype(int), zero_division=0)
    for t in thresholds
]
best_idx = int(np.argmax(val_f1s))
selected_threshold = float(thresholds[best_idx])

print("\nThreshold selection:")
print(f"Default threshold: {DEFAULT_THRESHOLD:.3f}")
print(f"Validation-selected threshold (max F1): {selected_threshold:.3f}")
print(f"Validation F1 at selected threshold: {val_f1s[best_idx]:.4f}")

def metrics(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "average_precision": float(average_precision_score(labels, probs)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall_sensitivity": float(recall_score(labels, preds, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

val_default = metrics(val_labels, val_probs, DEFAULT_THRESHOLD)
val_selected = metrics(val_labels, val_probs, selected_threshold)
test_default = metrics(test_labels, test_probs, DEFAULT_THRESHOLD)
test_selected = metrics(test_labels, test_probs, selected_threshold)

def print_metrics(title, m):
    print("\n" + title)
    print("-" * len(title))
    for key in ["threshold","accuracy","roc_auc","average_precision","precision","recall_sensitivity","specificity","f1"]:
        print(f"{key:20s}: {m[key]:.4f}")
    print(f"confusion matrix: TN={m['tn']}, FP={m['fp']}, FN={m['fn']}, TP={m['tp']}")

print_metrics("Validation metrics at threshold 0.5", val_default)
print_metrics("Validation metrics at selected threshold", val_selected)
print_metrics("Test metrics at threshold 0.5", test_default)
print_metrics("Test metrics at validation-selected threshold", test_selected)

# ROC
vfpr, vtpr, _ = roc_curve(val_labels, val_probs)
tfpr, ttpr, _ = roc_curve(test_labels, test_probs)
fig = plt.figure(figsize=(7,6))
plt.plot(vfpr, vtpr, label=f"Validation AUC = {val_default['roc_auc']:.3f}")
plt.plot(tfpr, ttpr, label=f"Test AUC = {test_default['roc_auc']:.3f}")
plt.plot([0,1],[0,1], linestyle="--", label="Chance")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Fine-Tuned ResNet50V2 ROC Curve"); plt.legend(); plt.tight_layout()
fig.savefig(FIGURES_DIR/"roc_curve.png", dpi=180, bbox_inches="tight"); plt.close(fig)

# PR
vp, vr, _ = precision_recall_curve(val_labels, val_probs)
tp, tr, _ = precision_recall_curve(test_labels, test_probs)
fig = plt.figure(figsize=(7,6))
plt.plot(vr, vp, label=f"Validation AP = {val_default['average_precision']:.3f}")
plt.plot(tr, tp, label=f"Test AP = {test_default['average_precision']:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Fine-Tuned ResNet50V2 Precision-Recall Curve"); plt.legend(); plt.tight_layout()
fig.savefig(FIGURES_DIR/"precision_recall_curve.png", dpi=180, bbox_inches="tight"); plt.close(fig)

def save_cm(m, title, filename):
    matrix = np.array([[m["tn"],m["fp"]],[m["fn"],m["tp"]]])
    fig = plt.figure(figsize=(6,5))
    plt.imshow(matrix)
    plt.title(title); plt.xlabel("Predicted label"); plt.ylabel("True label")
    plt.xticks([0,1],["Benign","Malignant"]); plt.yticks([0,1],["Benign","Malignant"])
    for i in range(2):
        for j in range(2):
            plt.text(j,i,str(matrix[i,j]),ha="center",va="center")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR/filename,dpi=180,bbox_inches="tight")
    plt.close(fig)

save_cm(test_default, "Fine-Tuned Test Confusion Matrix (Threshold = 0.5)", "confusion_matrix_test_threshold_0_5.png")
save_cm(test_selected, f"Fine-Tuned Test Confusion Matrix (Validation Threshold = {selected_threshold:.3f})", "confusion_matrix_test_selected_threshold.png")

# Probability distribution
fig = plt.figure(figsize=(8,6))
plt.hist(test_probs[test_labels==0], bins=25, alpha=0.6, label="Benign")
plt.hist(test_probs[test_labels==1], bins=25, alpha=0.6, label="Malignant")
plt.axvline(DEFAULT_THRESHOLD, linestyle="--", label="Threshold 0.5")
plt.axvline(selected_threshold, linestyle=":", label=f"Validation-selected threshold {selected_threshold:.3f}")
plt.xlabel("Predicted probability of malignancy"); plt.ylabel("Number of mammograms")
plt.title("Fine-Tuned Test Predicted Probability Distribution"); plt.legend(); plt.tight_layout()
fig.savefig(FIGURES_DIR/"test_probability_distribution.png",dpi=180,bbox_inches="tight"); plt.close(fig)

# Threshold selection
fig = plt.figure(figsize=(8,5))
plt.plot(thresholds, val_f1s)
plt.axvline(selected_threshold, linestyle="--", label=f"Selected = {selected_threshold:.3f}")
plt.xlabel("Classification Threshold"); plt.ylabel("Validation F1 Score")
plt.title("Fine-Tuned Model Validation Threshold Selection"); plt.legend(); plt.tight_layout()
fig.savefig(FIGURES_DIR/"validation_threshold_selection.png",dpi=180,bbox_inches="tight"); plt.close(fig)

# Per-image predictions
for frame, probs, name in [
    (validation_df, val_probs, "validation_predictions.csv"),
    (test_df, test_probs, "test_predictions.csv")
]:
    out = frame[["image_id","patient_id","binary_pathology","label","preprocessed_path"]].copy()
    out["predicted_probability"] = probs
    out["prediction_threshold_0_5"] = (probs >= DEFAULT_THRESHOLD).astype(int)
    out["prediction_selected_threshold"] = (probs >= selected_threshold).astype(int)
    out.to_csv(RESULTS_DIR/name, index=False)

summary = {
    "model": str(MODEL_PATH),
    "threshold_selection": {
        "method": "max_validation_f1",
        "default_threshold": DEFAULT_THRESHOLD,
        "selected_threshold": selected_threshold,
        "best_validation_f1": float(val_f1s[best_idx]),
    },
    "validation": {"threshold_0_5": val_default, "selected_threshold": val_selected},
    "test": {"threshold_0_5": test_default, "selected_threshold": test_selected},
}
with (RESULTS_DIR/"evaluation_metrics.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

pd.DataFrame([
    {"split":"validation","threshold_type":"0.5",**val_default},
    {"split":"validation","threshold_type":"validation_selected",**val_selected},
    {"split":"test","threshold_type":"0.5",**test_default},
    {"split":"test","threshold_type":"validation_selected",**test_selected},
]).to_csv(RESULTS_DIR/"evaluation_metrics.csv", index=False)

print("\n" + "="*72)
print("FINE-TUNED MODEL EVALUATION COMPLETE")
print("="*72)
print(f"\nValidation-selected threshold: {selected_threshold:.3f}")
print(f"Test ROC AUC: {test_default['roc_auc']:.4f}")
print(f"Test average precision: {test_default['average_precision']:.4f}")
print("\nTest threshold 0.5:")
print(f"  accuracy    = {test_default['accuracy']:.4f}")
print(f"  precision   = {test_default['precision']:.4f}")
print(f"  recall      = {test_default['recall_sensitivity']:.4f}")
print(f"  specificity = {test_default['specificity']:.4f}")
print(f"  F1          = {test_default['f1']:.4f}")
print("\nTest validation-selected threshold:")
print(f"  threshold   = {selected_threshold:.3f}")
print(f"  accuracy    = {test_selected['accuracy']:.4f}")
print(f"  precision   = {test_selected['precision']:.4f}")
print(f"  recall      = {test_selected['recall_sensitivity']:.4f}")
print(f"  specificity = {test_selected['specificity']:.4f}")
print(f"  F1          = {test_selected['f1']:.4f}")
print("\nSaved evaluation results:")
print(RESULTS_DIR.resolve())
