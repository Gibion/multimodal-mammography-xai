from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path.cwd()
MODEL_PATH = Path("models/resnet50v2_finetuned/best_model.keras")
TEST_PREDICTIONS = Path("results/resnet50v2_finetuned/evaluation/test_predictions.csv")
ROI_MANIFEST = Path("data/processed/cbis_ddsm/manifests/cbis_ddsm_roi_xai_manifest.csv")
OUTPUT_DIR = Path("results/resnet50v2_finetuned/gradcam_roi_alignment")
FIGURES_DIR = OUTPUT_DIR / "figures"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"

IMAGE_SIZE = 512
DEFAULT_THRESHOLD = 0.5
EPSILON = 1e-8
TOP_FRACTIONS = (0.10, 0.20, 0.30)

for d in (OUTPUT_DIR, FIGURES_DIR, HEATMAP_DIR):
    d.mkdir(parents=True, exist_ok=True)


def resolve_path(value):
    p = Path(str(value))
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def load_grayscale(path):
    with Image.open(path) as img:
        return np.asarray(img.convert("L"))


def load_mask(path):
    with Image.open(path) as img:
        return (np.asarray(img.convert("L")) > 0).astype(np.uint8)


def load_model_input(path):
    image = tf.io.decode_jpeg(tf.io.read_file(str(path)), channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32) * 255.0
    image = tf.ensure_shape(image, [IMAGE_SIZE, IMAGE_SIZE, 1])
    image = tf.image.grayscale_to_rgb(image)
    image = tf.ensure_shape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.keras.applications.resnet_v2.preprocess_input(image)
    return tf.expand_dims(image, axis=0)


def locate_backbone(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "resnet50v2" in layer.name.lower():
            return layer
    raise ValueError("Could not locate ResNet50V2 backbone.")


def locate_head(model):
    names = ("global_average_pooling", "dense_256", "dropout", "prediction")
    return {name: model.get_layer(name) for name in names}


def forward_head(feature_maps, head):
    x = head["global_average_pooling"](feature_maps)
    x = head["dense_256"](x)
    x = head["dropout"](x, training=False)
    return head["prediction"](x)


def gradcam(image_tensor, backbone, head):
    with tf.GradientTape() as tape:
        feature_maps = backbone(image_tensor, training=False)
        tape.watch(feature_maps)
        prediction = forward_head(feature_maps, head)
        score = prediction[:, 0]

    gradients = tape.gradient(score, feature_maps)
    if gradients is None:
        raise RuntimeError("Grad-CAM gradients are None.")

    weights = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap = tf.reduce_sum(
        feature_maps * weights[:, tf.newaxis, tf.newaxis, :],
        axis=-1
    )
    heatmap = tf.nn.relu(heatmap)[0]
    max_value = tf.reduce_max(heatmap)
    heatmap = tf.cond(
        max_value > 0,
        lambda: heatmap / max_value,
        lambda: tf.zeros_like(heatmap),
    )
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (IMAGE_SIZE, IMAGE_SIZE),
        method="bilinear",
    )
    heatmap = tf.squeeze(heatmap, axis=-1)
    return heatmap.numpy(), float(prediction[0, 0].numpy())


def union_masks(paths):
    union = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for path in paths:
        union = np.maximum(union, load_mask(Path(path)))
    return union


def pointing_game(heatmap, roi_mask):
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return int(roi_mask[y, x] > 0), int(x), int(y)


def energy_inside_roi(heatmap, roi_mask):
    total = float(heatmap.sum())
    if total <= EPSILON:
        return 0.0
    return float(heatmap[roi_mask > 0].sum() / total)


def activation_ratio(heatmap, roi_mask):
    inside = heatmap[roi_mask > 0]
    outside = heatmap[roi_mask == 0]
    inside_mean = float(inside.mean()) if inside.size else 0.0
    outside_mean = float(outside.mean()) if outside.size else 0.0
    ratio = inside_mean / (outside_mean + EPSILON)
    return ratio, inside_mean, outside_mean


def thresholded_iou(heatmap, roi_mask, top_fraction):
    threshold = float(np.percentile(heatmap, (1.0 - top_fraction) * 100.0))
    attention = heatmap >= threshold
    roi = roi_mask > 0
    intersection = np.logical_and(attention, roi).sum()
    union = np.logical_or(attention, roi).sum()
    return float(intersection / union) if union else 0.0


def outcome(true_label, pred_label):
    if true_label == 1 and pred_label == 1:
        return "TP"
    if true_label == 0 and pred_label == 0:
        return "TN"
    if true_label == 0 and pred_label == 1:
        return "FP"
    return "FN"


def lesion_group(values):
    values = {v.strip().lower() for v in str(values).split(",") if v.strip()}
    if values == {"mass"}:
        return "mass"
    if values == {"calcification"}:
        return "calcification"
    return "mixed"


print("=" * 72)
print("GRAD-CAM / ROI ALIGNMENT EVALUATION")
print("=" * 72)

model = tf.keras.models.load_model(MODEL_PATH)
backbone = locate_backbone(model)
head = locate_head(model)

print("\nBackbone output shape:")
print(backbone.output_shape)

pred_df = pd.read_csv(TEST_PREDICTIONS)
roi_df = pd.read_csv(ROI_MANIFEST)

print(f"\nTest prediction rows: {len(pred_df):,}")
print(f"ROI manifest rows: {len(roi_df):,}")
print("\nROI status:")
print(roi_df["roi_xai_status"].value_counts())

# Build one union-ROI row per full mammogram.
success = roi_df[roi_df["roi_xai_status"] == "success"].copy()
union_rows = []

for image_id, group in success.groupby("full_image_id", sort=False):
    roi_paths = [
        str(resolve_path(p))
        for p in group["roi_mask_512_path"]
        if pd.notna(p)
    ]
    lesion_types = sorted(set(group["abnormality type"].dropna().astype(str)))
    union_rows.append({
        "full_image_id": image_id,
        "roi_mask_paths": "||".join(roi_paths),
        "n_roi_masks": len(roi_paths),
        "lesion_types": ",".join(lesion_types),
    })

union_df = pd.DataFrame(union_rows)
union_df["lesion_group"] = union_df["lesion_types"].apply(lesion_group)

analysis_df = pred_df.merge(
    union_df,
    left_on="image_id",
    right_on="full_image_id",
    how="inner",
    validate="one_to_one",
)

analysis_df["predicted_label"] = (
    analysis_df["predicted_probability"] >= DEFAULT_THRESHOLD
).astype(int)
analysis_df["outcome"] = analysis_df.apply(
    lambda r: outcome(int(r["label"]), int(r["predicted_label"])),
    axis=1,
)

# Determine union ROI size and quartiles before Grad-CAM.
roi_fractions = []
for _, row in analysis_df.iterrows():
    mask = union_masks(str(row["roi_mask_paths"]).split("||"))
    roi_fractions.append(float((mask > 0).mean()))

analysis_df["union_roi_fraction"] = roi_fractions
q1 = float(analysis_df["union_roi_fraction"].quantile(0.25))
q3 = float(analysis_df["union_roi_fraction"].quantile(0.75))

def roi_size_group(value):
    if value <= q1:
        return "small"
    if value >= q3:
        return "large"
    return "medium"

analysis_df["roi_size_group"] = analysis_df["union_roi_fraction"].apply(
    roi_size_group
)

print(f"\nTest mammograms with usable ROI: {len(analysis_df):,}")
print("\nLesion groups:")
print(analysis_df["lesion_group"].value_counts())

records = []

print("\nEvaluating Grad-CAM localisation...")

for i, row in analysis_df.iterrows():
    image_path = resolve_path(row["preprocessed_path"])
    roi_mask = union_masks(str(row["roi_mask_paths"]).split("||"))

    heatmap, gc_prob = gradcam(
        load_model_input(image_path),
        backbone,
        head,
    )

    pg_hit, peak_x, peak_y = pointing_game(heatmap, roi_mask)
    energy = energy_inside_roi(heatmap, roi_mask)
    ratio, inside_mean, outside_mean = activation_ratio(heatmap, roi_mask)

    result = {
        "image_id": row["image_id"],
        "patient_id": row["patient_id"],
        "binary_pathology": row["binary_pathology"],
        "true_label": int(row["label"]),
        "predicted_label": int(row["predicted_label"]),
        "predicted_probability": float(row["predicted_probability"]),
        "gradcam_probability": gc_prob,
        "probability_difference": abs(
            float(row["predicted_probability"]) - gc_prob
        ),
        "outcome": row["outcome"],
        "lesion_group": row["lesion_group"],
        "lesion_types": row["lesion_types"],
        "n_roi_masks": int(row["n_roi_masks"]),
        "union_roi_fraction": float(row["union_roi_fraction"]),
        "roi_size_group": row["roi_size_group"],
        "pointing_game_hit": pg_hit,
        "gradcam_peak_x": peak_x,
        "gradcam_peak_y": peak_y,
        "energy_inside_roi": energy,
        "inside_mean_activation": inside_mean,
        "outside_mean_activation": outside_mean,
        "inside_outside_ratio": ratio,
        "preprocessed_path": str(image_path),
    }

    for frac in TOP_FRACTIONS:
        suffix = int(frac * 100)
        result[f"iou_top_{suffix}_percent"] = thresholded_iou(
            heatmap,
            roi_mask,
            frac,
        )

    heatmap_path = HEATMAP_DIR / f"{row['image_id']}.npy"
    np.save(heatmap_path, heatmap.astype(np.float32))
    result["heatmap_npy_path"] = str(heatmap_path.resolve())

    records.append(result)

    if (i + 1) % 25 == 0 or (i + 1) == len(analysis_df):
        print(f"  processed {i + 1:,} / {len(analysis_df):,}")

metrics_df = pd.DataFrame(records)
metrics_csv = OUTPUT_DIR / "gradcam_roi_metrics_per_image.csv"
metrics_df.to_csv(metrics_csv, index=False)

metric_columns = [
    "pointing_game_hit",
    "energy_inside_roi",
    "inside_outside_ratio",
    "iou_top_10_percent",
    "iou_top_20_percent",
    "iou_top_30_percent",
]

overall = pd.DataFrame({
    "metric": metric_columns,
    "mean": [metrics_df[c].mean() for c in metric_columns],
    "median": [metrics_df[c].median() for c in metric_columns],
})
overall.to_csv(OUTPUT_DIR / "summary_overall.csv", index=False)

for group_col, filename in [
    ("lesion_group", "summary_by_lesion_group.csv"),
    ("outcome", "summary_by_prediction_outcome.csv"),
    ("roi_size_group", "summary_by_roi_size.csv"),
]:
    summary = metrics_df.groupby(group_col)[metric_columns].agg(
        ["count", "mean", "median"]
    )
    summary.to_csv(OUTPUT_DIR / filename)

# Lesion-level secondary analysis using already saved image-level heatmaps.
lesion_test = success.merge(
    pred_df,
    left_on="full_image_id",
    right_on="image_id",
    how="inner",
)

lesion_records = []

for _, row in lesion_test.iterrows():
    heatmap = np.load(HEATMAP_DIR / f"{row['image_id']}.npy")
    roi_mask = load_mask(resolve_path(row["roi_mask_512_path"]))

    pg_hit, peak_x, peak_y = pointing_game(heatmap, roi_mask)
    energy = energy_inside_roi(heatmap, roi_mask)
    ratio, inside_mean, outside_mean = activation_ratio(heatmap, roi_mask)

    lesion_record = {
        "image_id": row["image_id"],
        "patient_id": row["patient_id_x"]
        if "patient_id_x" in row.index
        else row["patient_id"],
        "abnormality_id": row.get("abnormality id", np.nan),
        "abnormality_type": row.get("abnormality type", ""),
        "pathology": row["pathology"],
        "predicted_probability": float(row["predicted_probability"]),
        "pointing_game_hit": pg_hit,
        "gradcam_peak_x": peak_x,
        "gradcam_peak_y": peak_y,
        "energy_inside_roi": energy,
        "inside_mean_activation": inside_mean,
        "outside_mean_activation": outside_mean,
        "inside_outside_ratio": ratio,
        "roi_foreground_fraction_512": float(
            row["roi_foreground_fraction_512"]
        ),
    }

    for frac in TOP_FRACTIONS:
        suffix = int(frac * 100)
        lesion_record[f"iou_top_{suffix}_percent"] = thresholded_iou(
            heatmap,
            roi_mask,
            frac,
        )

    lesion_records.append(lesion_record)

pd.DataFrame(lesion_records).to_csv(
    OUTPUT_DIR / "gradcam_roi_metrics_per_lesion.csv",
    index=False,
)

# Representative high/low localisation figures.
def save_overlay(row, label):
    image = load_grayscale(resolve_path(row["preprocessed_path"]))
    heatmap = np.load(row["heatmap_npy_path"])

    union_row = analysis_df[
        analysis_df["image_id"] == row["image_id"]
    ].iloc[0]
    roi_mask = union_masks(str(union_row["roi_mask_paths"]).split("||"))

    fig = plt.figure(figsize=(7, 7))
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.imshow(heatmap, alpha=0.40, vmin=0, vmax=1)
    plt.contour((roi_mask > 0).astype(np.uint8), levels=[0.5], linewidths=1.5)
    plt.scatter(
        [row["gradcam_peak_x"]],
        [row["gradcam_peak_y"]],
        marker="x",
        s=60,
    )
    plt.title(
        f"{label} | {row['outcome']} | {row['lesion_group']} | "
        f"PG={row['pointing_game_hit']} | "
        f"Energy={row['energy_inside_roi']:.3f} | "
        f"Ratio={row['inside_outside_ratio']:.2f}"
    )
    plt.axis("off")
    plt.tight_layout()

    output_path = FIGURES_DIR / f"{label}_{row['image_id']}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

high = metrics_df.sort_values(
    ["pointing_game_hit", "energy_inside_roi", "inside_outside_ratio"],
    ascending=[False, False, False],
).head(4)

low = metrics_df.sort_values(
    ["pointing_game_hit", "energy_inside_roi", "inside_outside_ratio"],
    ascending=[True, True, True],
).head(4)

for _, row in high.iterrows():
    save_overlay(row, "high_localisation")

for _, row in low.iterrows():
    save_overlay(row, "low_localisation")

# Distribution figure.
fig = plt.figure(figsize=(8, 5))
for group in ("TP", "TN", "FP", "FN"):
    values = metrics_df.loc[
        metrics_df["outcome"] == group,
        "energy_inside_roi",
    ]
    if len(values):
        plt.hist(values, bins=20, alpha=0.45, label=group)

plt.xlabel("Grad-CAM Energy Inside ROI")
plt.ylabel("Number of Mammograms")
plt.title("Grad-CAM Energy Inside ROI by Prediction Outcome")
plt.legend()
plt.tight_layout()
fig.savefig(
    FIGURES_DIR / "energy_inside_roi_by_outcome.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close(fig)

summary = {
    "analysis_unit_primary": "full_mammogram_union_roi",
    "number_of_test_mammograms_with_roi": int(len(metrics_df)),
    "pointing_game_accuracy": float(metrics_df["pointing_game_hit"].mean()),
    "mean_energy_inside_roi": float(metrics_df["energy_inside_roi"].mean()),
    "median_energy_inside_roi": float(metrics_df["energy_inside_roi"].median()),
    "mean_inside_outside_ratio": float(
        metrics_df["inside_outside_ratio"].mean()
    ),
    "median_inside_outside_ratio": float(
        metrics_df["inside_outside_ratio"].median()
    ),
    "mean_iou_top_10_percent": float(
        metrics_df["iou_top_10_percent"].mean()
    ),
    "mean_iou_top_20_percent": float(
        metrics_df["iou_top_20_percent"].mean()
    ),
    "mean_iou_top_30_percent": float(
        metrics_df["iou_top_30_percent"].mean()
    ),
    "roi_size_quartiles": {"q1": q1, "q3": q3},
}

with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 72)
print("GRAD-CAM / ROI ALIGNMENT COMPLETE")
print("=" * 72)

print(f"\nTest mammograms evaluated: {len(metrics_df):,}")
print(
    f"Pointing-game accuracy: "
    f"{metrics_df['pointing_game_hit'].mean():.4f}"
)
print(
    f"Mean energy inside ROI: "
    f"{metrics_df['energy_inside_roi'].mean():.4f}"
)
print(
    f"Median energy inside ROI: "
    f"{metrics_df['energy_inside_roi'].median():.4f}"
)
print(
    f"Mean inside/outside activation ratio: "
    f"{metrics_df['inside_outside_ratio'].mean():.4f}"
)
print(
    f"Median inside/outside activation ratio: "
    f"{metrics_df['inside_outside_ratio'].median():.4f}"
)
print(
    f"Mean IoU, top 10% Grad-CAM: "
    f"{metrics_df['iou_top_10_percent'].mean():.4f}"
)
print(
    f"Mean IoU, top 20% Grad-CAM: "
    f"{metrics_df['iou_top_20_percent'].mean():.4f}"
)
print(
    f"Mean IoU, top 30% Grad-CAM: "
    f"{metrics_df['iou_top_30_percent'].mean():.4f}"
)

print("\nPrediction outcomes:")
print(metrics_df["outcome"].value_counts())

print("\nLesion groups:")
print(metrics_df["lesion_group"].value_counts())

print("\nROI size groups:")
print(metrics_df["roi_size_group"].value_counts())

print("\nSaved per-image metrics:")
print(metrics_csv.resolve())

print("\nSaved results directory:")
print(OUTPUT_DIR.resolve())
