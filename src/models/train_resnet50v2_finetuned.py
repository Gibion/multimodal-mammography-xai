from pathlib import Path
import json, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path.cwd()
MANIFEST_FILE = Path("data/processed/cbis_ddsm/manifests/cbis_ddsm_full_image_manifest_preprocessed_v4.csv")
BASELINE_MODEL_PATH = Path("models/resnet50v2_baseline/best_model.keras")
MODEL_DIR = Path("models/resnet50v2_finetuned")
RESULTS_DIR = Path("results/resnet50v2_finetuned")
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"

IMAGE_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 20
RANDOM_SEED = 42
FINE_TUNE_LEARNING_RATE = 1e-5
EARLY_STOPPING_PATIENCE = 5
LR_PATIENCE = 2
LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-7
FINE_TUNE_PREFIX = "conv5_"
AUTOTUNE = tf.data.AUTOTUNE

tf.keras.utils.set_random_seed(RANDOM_SEED)

for d in [MODEL_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST_FILE)
required = ["preprocessed_path","preprocessing_status","binary_pathology","split","patient_id","image_id"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError("Missing columns: " + ", ".join(missing))

df = df[df["preprocessing_status"] == "success"].copy()

def resolve_path(value):
    p = Path(str(value))
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p.resolve())

df["resolved_path"] = df["preprocessed_path"].apply(resolve_path)
exists = df["resolved_path"].apply(os.path.exists)

print("=" * 72)
print("RESNET50V2 PARTIAL FINE-TUNING")
print("=" * 72)
print(f"\nManifest rows: {len(df):,}")
print(f"Source images found: {exists.sum():,} / {len(exists):,}")

if not exists.all():
    raise FileNotFoundError(f"{(~exists).sum():,} source images are missing.")

label_map = {"BENIGN": 0, "MALIGNANT": 1}
df["label"] = df["binary_pathology"].astype(str).str.upper().map(label_map)
if df["label"].isna().any():
    raise ValueError("Unexpected pathology labels.")
df["label"] = df["label"].astype(np.float32)

train_df = df[df["split"] == "train"].copy()
validation_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

print("\nSplit counts:")
print(df["split"].value_counts())
print("\nClass counts by split:")
print(pd.crosstab(df["split"], df["binary_pathology"]))

def load_image(path, label):
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32) * 255.0
    image = tf.ensure_shape(image, [IMAGE_SIZE, IMAGE_SIZE, 1])
    image = tf.image.grayscale_to_rgb(image)
    image = tf.ensure_shape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    return image, label

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(7.0/360.0, fill_mode="constant", fill_value=0.0, seed=RANDOM_SEED),
    tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="constant", fill_value=0.0, seed=RANDOM_SEED+1),
    tf.keras.layers.RandomZoom(height_factor=(-0.08,0.08), width_factor=(-0.08,0.08), fill_mode="constant", fill_value=0.0, seed=RANDOM_SEED+2),
    tf.keras.layers.RandomContrast(0.08, seed=RANDOM_SEED+3),
], name="training_augmentation")
augmentation.build((None, IMAGE_SIZE, IMAGE_SIZE, 3))

def augment_image(image, label):
    image = tf.expand_dims(image, 0)
    image = augmentation(image, training=True)
    image = tf.squeeze(image, 0)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label

def resnet_preprocess(image, label):
    return tf.keras.applications.resnet_v2.preprocess_input(image), label

def build_dataset(frame, training=False):
    ds = tf.data.Dataset.from_tensor_slices((
        frame["resolved_path"].to_numpy(),
        frame["label"].to_numpy(dtype=np.float32)
    ))
    if training:
        ds = ds.shuffle(len(frame), seed=RANDOM_SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment_image, num_parallel_calls=AUTOTUNE)
    ds = ds.map(resnet_preprocess, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_dataset = build_dataset(train_df, True)
validation_dataset = build_dataset(validation_df, False)
test_dataset = build_dataset(test_df, False)

if not BASELINE_MODEL_PATH.exists():
    raise FileNotFoundError(f"Baseline model not found: {BASELINE_MODEL_PATH.resolve()}")

print("\nLoading baseline model:")
print(BASELINE_MODEL_PATH.resolve())
model = tf.keras.models.load_model(BASELINE_MODEL_PATH)

base_model = None
for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and "resnet50v2" in layer.name.lower():
        base_model = layer
        break
if base_model is None:
    raise ValueError("Could not locate ResNet50V2 backbone.")

# Freeze all backbone layers, then unfreeze conv5_x only.
for layer in base_model.layers:
    layer.trainable = False
for layer in base_model.layers:
    if layer.name.startswith(FINE_TUNE_PREFIX):
        layer.trainable = True
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

trainable_backbone_layers = [l for l in base_model.layers if l.trainable]
frozen_backbone_layers = [l for l in base_model.layers if not l.trainable]

print("\nBackbone:", base_model.name)
print("Total backbone layers:", len(base_model.layers))
print("Trainable backbone layers:", len(trainable_backbone_layers))
print("Frozen backbone layers:", len(frozen_backbone_layers))
print("Fine-tune prefix:", FINE_TUNE_PREFIX)

print("\nFirst trainable backbone layers:")
for layer in trainable_backbone_layers[:8]:
    print(" ", layer.name)
print("\nLast trainable backbone layers:")
for layer in trainable_backbone_layers[-8:]:
    print(" ", layer.name)

model.compile(
    optimizer=tf.keras.optimizers.Adam(FINE_TUNE_LEARNING_RATE),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]
)

with (RESULTS_DIR / "model_summary.txt").open("w", encoding="utf-8") as f:
    model.summary(print_fn=lambda line: f.write(line + "\n"))

pd.DataFrame([
    {"layer_name": l.name, "layer_type": l.__class__.__name__, "trainable": bool(l.trainable)}
    for l in base_model.layers
]).to_csv(RESULTS_DIR / "backbone_trainability.csv", index=False)

print("\nEvaluating starting baseline checkpoint on validation set...")
starting_validation_results = model.evaluate(validation_dataset, verbose=1, return_dict=True)

best_model_path = MODEL_DIR / "best_model.keras"
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(best_model_path, monitor="val_auc", mode="max", save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LEARNING_RATE, verbose=1),
    tf.keras.callbacks.CSVLogger(LOGS_DIR / "training_log.csv", append=False),
]

print("\nStarting partial fine-tuning...")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

final_model_path = MODEL_DIR / "final_model.keras"
model.save(final_model_path)

history_df = pd.DataFrame(history.history)
history_df.to_csv(RESULTS_DIR / "training_history.csv", index=False)

def save_metric_plot(metric, title, filename):
    val_metric = "val_" + metric
    if metric not in history_df.columns:
        return
    fig = plt.figure(figsize=(7,5))
    plt.plot(history_df.index + 1, history_df[metric], label="Training")
    if val_metric in history_df.columns:
        plt.plot(history_df.index + 1, history_df[val_metric], label="Validation")
    plt.xlabel("Fine-Tuning Epoch")
    plt.ylabel(metric.replace("_"," ").title())
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)

save_metric_plot("loss", "Fine-Tuning Training and Validation Loss", "loss_curve.png")
save_metric_plot("auc", "Fine-Tuning Training and Validation AUC", "auc_curve.png")
save_metric_plot("accuracy", "Fine-Tuning Training and Validation Accuracy", "accuracy_curve.png")

print("\nEvaluating fine-tuned model on validation set...")
validation_results = model.evaluate(validation_dataset, verbose=1, return_dict=True)
print("\nValidation results:")
for k,v in validation_results.items():
    print(f"{k:12s}: {v:.4f}")

print("\nEvaluating on held-out test set...")
test_results = model.evaluate(test_dataset, verbose=1, return_dict=True)
print("\nTest results:")
for k,v in test_results.items():
    print(f"{k:12s}: {v:.4f}")

metrics = {
    "configuration": {
        "starting_model": str(BASELINE_MODEL_PATH),
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_requested": EPOCHS,
        "epochs_completed": len(history_df),
        "fine_tune_learning_rate": FINE_TUNE_LEARNING_RATE,
        "fine_tune_prefix": FINE_TUNE_PREFIX,
        "trainable_backbone_layers": len(trainable_backbone_layers),
        "frozen_backbone_layers": len(frozen_backbone_layers),
        "batch_normalization_frozen": True,
    },
    "starting_validation": {k: float(v) for k,v in starting_validation_results.items()},
    "final_validation": {k: float(v) for k,v in validation_results.items()},
    "test": {k: float(v) for k,v in test_results.items()},
}

with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

best_epoch = int(history_df["val_auc"].idxmax() + 1)
best_val_auc = float(history_df["val_auc"].max())
starting_val_auc = float(starting_validation_results["auc"])

print("\n" + "=" * 72)
print("FINE-TUNING COMPLETE")
print("=" * 72)
print(f"\nEpochs completed: {len(history_df)}")
print(f"Starting validation AUC: {starting_val_auc:.4f}")
print(f"Best fine-tuned validation AUC: {best_val_auc:.4f}")
print(f"Validation AUC change: {best_val_auc - starting_val_auc:+.4f}")
print(f"Best fine-tuning epoch: {best_epoch}")
print(f"Test AUC: {test_results['auc']:.4f}")
print("\nSaved best model:")
print(best_model_path.resolve())
print("\nSaved final model:")
print(final_model_path.resolve())
print("\nSaved results:")
print(RESULTS_DIR.resolve())
