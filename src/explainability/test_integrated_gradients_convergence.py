"""
test_integrated_gradients_convergence.py

Small stratified Integrated Gradients convergence experiment for the
fine-tuned ResNet50V2 model.

Evaluates:
- completeness absolute error;
- completeness relative error;
- map correlation against the 400-step reference;
- mean absolute map difference against the 400-step reference.

Sample:
- 4 TP
- 4 TN
- 4 FP
- 4 FN
= 16 mammograms total.
"""

from pathlib import Path
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


PROJECT_ROOT = Path.cwd()

MODEL_FILE = Path(
    "models/resnet50v2_finetuned/best_model.keras"
)

TEST_PREDICTIONS_FILE = Path(
    "results/resnet50v2_finetuned/evaluation/test_predictions.csv"
)

ROI_XAI_MANIFEST_FILE = Path(
    "data/processed/cbis_ddsm/manifests/"
    "cbis_ddsm_roi_xai_manifest.csv"
)

OUTPUT_DIR = Path(
    "results/resnet50v2_finetuned/"
    "integrated_gradients_convergence"
)

RESULTS_CSV = (
    OUTPUT_DIR
    / "integrated_gradients_convergence_per_case.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "integrated_gradients_convergence_summary.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "integrated_gradients_convergence_summary.json"
)

FIGURE_COMPLETENESS = (
    OUTPUT_DIR
    / "ig_completeness_convergence.png"
)

FIGURE_SIMILARITY = (
    OUTPUT_DIR
    / "ig_map_similarity_convergence.png"
)

IMAGE_SIZE = (512, 512)
CLASSIFICATION_THRESHOLD = 0.5
SAMPLES_PER_OUTCOME = 4
STEP_COUNTS = [25, 50, 100, 200, 400]
REFERENCE_STEPS = 400
INTEGRATION_BATCH_SIZE = 20
BASELINE_VALUE = -1.0
RANDOM_SEED = 42
EPSILON = 1e-10


def resolve_path(value):
    if pd.isna(value):
        return None

    path = Path(str(value))

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def find_column(frame, candidates):
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    raise ValueError(
        "Could not find any of these columns: "
        + ", ".join(candidates)
    )


def pathology_to_int(value):
    text = str(value).strip().upper()

    if text == "MALIGNANT":
        return 1

    if text == "BENIGN":
        return 0

    if text in {"0", "1"}:
        return int(text)

    raise ValueError(
        f"Unexpected pathology label: {value}"
    )


def classify_outcome(
    label,
    probability,
    threshold=0.5,
):
    prediction = int(probability >= threshold)

    if label == 1 and prediction == 1:
        return "TP"

    if label == 0 and prediction == 0:
        return "TN"

    if label == 0 and prediction == 1:
        return "FP"

    return "FN"


def load_model_input_image(path):
    image_bytes = tf.io.read_file(str(path))

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
        IMAGE_SIZE,
        method="bilinear",
    )

    model_input = (
        tf.keras.applications.resnet_v2.preprocess_input(
            image * 255.0
        )
    )

    return tf.expand_dims(
        model_input,
        axis=0,
    )


def make_baseline_like(image_batch):
    return (
        tf.ones_like(
            image_batch,
            dtype=tf.float32,
        )
        * BASELINE_VALUE
    )


def interpolate_inputs(
    baseline,
    image,
    alphas,
):
    alphas = alphas[
        :,
        tf.newaxis,
        tf.newaxis,
        tf.newaxis,
    ]

    return (
        baseline[0][tf.newaxis, ...]
        +
        alphas
        * (
            image[0][tf.newaxis, ...]
            -
            baseline[0][tf.newaxis, ...]
        )
    )


def compute_input_gradients(
    model,
    interpolated_batch,
):
    with tf.GradientTape() as tape:
        tape.watch(interpolated_batch)

        probabilities = model(
            interpolated_batch,
            training=False,
        )[:, 0]

    gradients = tape.gradient(
        probabilities,
        interpolated_batch,
    )

    if gradients is None:
        raise RuntimeError(
            "Input-gradient computation returned None."
        )

    return gradients


def compute_integrated_gradients(
    model,
    image_batch,
    n_steps,
):
    baseline = make_baseline_like(
        image_batch
    )

    baseline_probability = float(
        model(
            baseline,
            training=False,
        ).numpy()[0, 0]
    )

    image_probability = float(
        model(
            image_batch,
            training=False,
        ).numpy()[0, 0]
    )

    alphas = tf.linspace(
        0.0,
        1.0,
        n_steps + 1,
    )

    gradient_chunks = []

    for start in range(
        0,
        n_steps + 1,
        INTEGRATION_BATCH_SIZE,
    ):
        stop = min(
            start + INTEGRATION_BATCH_SIZE,
            n_steps + 1,
        )

        interpolated = interpolate_inputs(
            baseline,
            image_batch,
            alphas[start:stop],
        )

        gradients = compute_input_gradients(
            model,
            interpolated,
        )

        gradient_chunks.append(
            gradients
        )

    gradients = tf.concat(
        gradient_chunks,
        axis=0,
    )

    trapezoids = (
        gradients[:-1]
        +
        gradients[1:]
    ) / 2.0

    average_gradients = tf.reduce_mean(
        trapezoids,
        axis=0,
    )

    input_difference = (
        image_batch[0]
        -
        baseline[0]
    )

    attributions = (
        input_difference
        *
        average_gradients
    )

    attribution_sum = float(
        tf.reduce_sum(
            attributions
        ).numpy()
    )

    prediction_difference = (
        image_probability
        -
        baseline_probability
    )

    completeness_delta = (
        attribution_sum
        -
        prediction_difference
    )

    absolute_error = abs(
        completeness_delta
    )

    relative_error = (
        absolute_error
        /
        max(
            abs(prediction_difference),
            EPSILON,
        )
    )

    return {
        "attributions": (
            attributions
            .numpy()
            .astype(np.float32)
        ),
        "baseline_probability": baseline_probability,
        "image_probability": image_probability,
        "prediction_difference": prediction_difference,
        "attribution_sum": attribution_sum,
        "completeness_delta": completeness_delta,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
    }


def collapse_positive_attributions(
    attributions
):
    positive = np.maximum(
        attributions,
        0.0,
    )

    heatmap = np.sum(
        positive,
        axis=-1,
    )

    maximum = float(
        heatmap.max()
    )

    if maximum > EPSILON:
        heatmap = heatmap / maximum
    else:
        heatmap = np.zeros_like(
            heatmap,
            dtype=np.float32,
        )

    return heatmap.astype(
        np.float32
    )


def pearson_map_correlation(
    map_a,
    map_b,
):
    a = map_a.ravel().astype(float)
    b = map_b.ravel().astype(float)

    if (
        np.std(a) <= EPSILON
        or np.std(b) <= EPSILON
    ):
        return np.nan

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


def mean_absolute_map_difference(
    map_a,
    map_b,
):
    return float(
        np.mean(
            np.abs(
                map_a
                -
                map_b
            )
        )
    )


for path in [
    MODEL_FILE,
    TEST_PREDICTIONS_FILE,
    ROI_XAI_MANIFEST_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{path.resolve()}"
        )


predictions_df = pd.read_csv(
    TEST_PREDICTIONS_FILE
)

roi_df = pd.read_csv(
    ROI_XAI_MANIFEST_FILE
)

prediction_id_column = find_column(
    predictions_df,
    [
        "full_image_id",
        "image_id",
    ],
)

prediction_probability_column = find_column(
    predictions_df,
    [
        "predicted_probability",
        "probability",
        "prediction",
    ],
)

prediction_image_path_column = find_column(
    predictions_df,
    [
        "preprocessed_path",
        "full_preprocessed_path",
        "image_path",
    ],
)

prediction_pathology_column = find_column(
    predictions_df,
    [
        "binary_pathology",
        "pathology",
    ],
)

prediction_patient_column = find_column(
    predictions_df,
    [
        "patient_id",
    ],
)

roi_status_column = find_column(
    roi_df,
    [
        "roi_xai_status",
    ],
)

roi_split_column = find_column(
    roi_df,
    [
        "full_split",
        "split",
    ],
)

roi_id_column = find_column(
    roi_df,
    [
        "full_image_id",
        "image_id",
    ],
)

successful_test_roi = roi_df[
    (
        roi_df[
            roi_status_column
        ] == "success"
    )
    &
    (
        roi_df[
            roi_split_column
        ] == "test"
    )
].copy()

roi_test_ids = set(
    successful_test_roi[
        roi_id_column
    ].astype(str)
)

cohort = predictions_df[
    predictions_df[
        prediction_id_column
    ].astype(str).isin(
        roi_test_ids
    )
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

cohort[
    "full_image_id"
] = cohort[
    "full_image_id"
].astype(str)

cohort = cohort.drop_duplicates(
    subset=[
        "full_image_id"
    ]
).reset_index(
    drop=True
)

if len(cohort) != 414:
    raise ValueError(
        "Expected 414 ROI-evaluable test mammograms, "
        f"but found {len(cohort)}."
    )

cohort[
    "label"
] = cohort[
    "binary_pathology"
].apply(
    pathology_to_int
)

cohort[
    "outcome"
] = [
    classify_outcome(
        label,
        probability,
        CLASSIFICATION_THRESHOLD,
    )
    for label, probability in zip(
        cohort[
            "label"
        ],
        cohort[
            "predicted_probability"
        ],
    )
]

cohort[
    "resolved_preprocessed_path"
] = cohort[
    "preprocessed_path"
].apply(
    resolve_path
)

if not cohort[
    "resolved_preprocessed_path"
].apply(
    lambda path: (
        path is not None
        and path.exists()
    )
).all():
    raise FileNotFoundError(
        "One or more preprocessed mammograms are missing."
    )


sample_frames = []

outcomes = [
    "TP",
    "TN",
    "FP",
    "FN",
]

for index, outcome in enumerate(
    outcomes
):
    subset = cohort[
        cohort[
            "outcome"
        ] == outcome
    ]

    if len(subset) < SAMPLES_PER_OUTCOME:
        raise ValueError(
            f"Not enough {outcome} examples."
        )

    sampled = subset.sample(
        n=SAMPLES_PER_OUTCOME,
        random_state=(
            RANDOM_SEED
            + index
        ),
    )

    sample_frames.append(
        sampled
    )

sample_df = pd.concat(
    sample_frames,
    ignore_index=True,
)


print("=" * 72)
print("INTEGRATED GRADIENTS CONVERGENCE TEST")
print("=" * 72)

print("\nLoading model:")
print(MODEL_FILE.resolve())

model = tf.keras.models.load_model(
    MODEL_FILE
)

print(
    f"\nROI-evaluable test cohort: "
    f"{len(cohort)}"
)

print(
    f"Convergence sample size: "
    f"{len(sample_df)}"
)

print("\nSample counts by outcome:")
print(
    sample_df[
        "outcome"
    ].value_counts()
)

print("\nIntegration step counts:")
print(STEP_COUNTS)

print(
    f"\nReference map: "
    f"{REFERENCE_STEPS} steps"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

records = []

start_time = time.time()

for case_index, row in sample_df.iterrows():
    image_batch = load_model_input_image(
        row[
            "resolved_preprocessed_path"
        ]
    )

    saved_probability = float(
        row[
            "predicted_probability"
        ]
    )

    print(
        f"\nCase {case_index + 1}/"
        f"{len(sample_df)}: "
        f"{row['patient_id']} "
        f"{row['outcome']}"
    )

    case_results = {}

    for steps in STEP_COUNTS:
        print(
            f"  IG steps = {steps}"
        )

        result = compute_integrated_gradients(
            model,
            image_batch,
            steps,
        )

        heatmap = collapse_positive_attributions(
            result[
                "attributions"
            ]
        )

        case_results[
            steps
        ] = {
            **result,
            "heatmap": heatmap,
        }

    reference_map = case_results[
        REFERENCE_STEPS
    ][
        "heatmap"
    ]

    for steps in STEP_COUNTS:
        result = case_results[
            steps
        ]

        prediction_absolute_difference = abs(
            result[
                "image_probability"
            ]
            -
            saved_probability
        )

        if steps == REFERENCE_STEPS:
            map_correlation = 1.0
            map_mae = 0.0
        else:
            map_correlation = (
                pearson_map_correlation(
                    result[
                        "heatmap"
                    ],
                    reference_map,
                )
            )

            map_mae = (
                mean_absolute_map_difference(
                    result[
                        "heatmap"
                    ],
                    reference_map,
                )
            )

        records.append({
            "full_image_id": row[
                "full_image_id"
            ],
            "patient_id": row[
                "patient_id"
            ],
            "binary_pathology": row[
                "binary_pathology"
            ],
            "outcome": row[
                "outcome"
            ],
            "saved_probability": (
                saved_probability
            ),
            "generated_probability": (
                result[
                    "image_probability"
                ]
            ),
            "prediction_absolute_difference": (
                prediction_absolute_difference
            ),
            "baseline_probability": (
                result[
                    "baseline_probability"
                ]
            ),
            "prediction_minus_baseline": (
                result[
                    "prediction_difference"
                ]
            ),
            "integration_steps": int(
                steps
            ),
            "attribution_sum_signed": (
                result[
                    "attribution_sum"
                ]
            ),
            "completeness_delta_signed": (
                result[
                    "completeness_delta"
                ]
            ),
            "completeness_absolute_error": (
                result[
                    "absolute_error"
                ]
            ),
            "completeness_relative_error": (
                result[
                    "relative_error"
                ]
            ),
            "map_correlation_vs_400": (
                map_correlation
            ),
            "map_mae_vs_400": (
                map_mae
            ),
        })


results_df = pd.DataFrame(
    records
)

results_df.to_csv(
    RESULTS_CSV,
    index=False,
)

summary_df = (
    results_df.groupby(
        "integration_steps",
        as_index=False,
    )
    .agg(
        n=(
            "full_image_id",
            "count",
        ),
        mean_abs_completeness_error=(
            "completeness_absolute_error",
            "mean",
        ),
        median_abs_completeness_error=(
            "completeness_absolute_error",
            "median",
        ),
        mean_relative_completeness_error=(
            "completeness_relative_error",
            "mean",
        ),
        median_relative_completeness_error=(
            "completeness_relative_error",
            "median",
        ),
        mean_map_correlation_vs_400=(
            "map_correlation_vs_400",
            "mean",
        ),
        median_map_correlation_vs_400=(
            "map_correlation_vs_400",
            "median",
        ),
        mean_map_mae_vs_400=(
            "map_mae_vs_400",
            "mean",
        ),
        median_map_mae_vs_400=(
            "map_mae_vs_400",
            "median",
        ),
    )
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False,
)

elapsed_seconds = (
    time.time()
    -
    start_time
)

with SUMMARY_JSON.open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        {
            "sample_size": int(
                len(
                    sample_df
                )
            ),
            "samples_per_outcome": int(
                SAMPLES_PER_OUTCOME
            ),
            "step_counts": STEP_COUNTS,
            "reference_steps": int(
                REFERENCE_STEPS
            ),
            "integration_batch_size": int(
                INTEGRATION_BATCH_SIZE
            ),
            "baseline_value_model_input_space": float(
                BASELINE_VALUE
            ),
            "elapsed_seconds": float(
                elapsed_seconds
            ),
            "summary": (
                summary_df.to_dict(
                    orient="records"
                )
            ),
        },
        handle,
        indent=2,
    )


fig = plt.figure(
    figsize=(8, 6)
)

plt.plot(
    summary_df[
        "integration_steps"
    ],
    summary_df[
        "median_abs_completeness_error"
    ],
    marker="o",
    label="Median absolute error",
)

plt.plot(
    summary_df[
        "integration_steps"
    ],
    summary_df[
        "mean_abs_completeness_error"
    ],
    marker="o",
    label="Mean absolute error",
)

plt.xlabel(
    "Integrated Gradients steps"
)
plt.ylabel(
    "Completeness error"
)
plt.title(
    "Integrated Gradients Completeness Convergence"
)
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

fig.savefig(
    FIGURE_COMPLETENESS,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


fig = plt.figure(
    figsize=(8, 6)
)

plt.plot(
    summary_df[
        "integration_steps"
    ],
    summary_df[
        "median_map_correlation_vs_400"
    ],
    marker="o",
    label="Median correlation vs 400-step map",
)

plt.plot(
    summary_df[
        "integration_steps"
    ],
    summary_df[
        "mean_map_correlation_vs_400"
    ],
    marker="o",
    label="Mean correlation vs 400-step map",
)

plt.xlabel(
    "Integrated Gradients steps"
)
plt.ylabel(
    "Pearson map correlation"
)
plt.ylim(
    0,
    1.05,
)
plt.title(
    "Integrated Gradients Spatial Attribution Convergence"
)
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

fig.savefig(
    FIGURE_SIMILARITY,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print("\n")
print("=" * 72)
print("INTEGRATED GRADIENTS CONVERGENCE COMPLETE")
print("=" * 72)

print("\nSummary by integration step:")

print(
    summary_df.to_string(
        index=False
    )
)

print(
    f"\nElapsed time: "
    f"{elapsed_seconds / 60.0:.1f} minutes"
)

print("\nSaved per-case results:")
print(RESULTS_CSV.resolve())

print("\nSaved convergence summary:")
print(SUMMARY_CSV.resolve())

print("\nSaved figures:")
print(FIGURE_COMPLETENESS.resolve())
print(FIGURE_SIMILARITY.resolve())
