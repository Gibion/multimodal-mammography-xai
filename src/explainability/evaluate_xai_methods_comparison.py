"""
evaluate_xai_methods_comparison.py

Final controlled RQ2 comparison for:
    1. Grad-CAM
    2. Grad-CAM++
    3. Integrated Gradients (200 steps)

All methods are compared on the same 414 ROI-evaluable CBIS-DDSM
test mammograms.

Grad-CAM:
    Reuses the previously generated per-image ROI localisation metrics.

Grad-CAM++ and Integrated Gradients:
    Recompute localisation metrics from their saved raw 512 x 512
    attribution maps against the union of all usable lesion ROI masks
    belonging to each mammogram.

Metrics:
    - Pointing-game hit
    - Energy fraction inside ROI
    - Inside/outside activation ratio
    - IoU for top 10%, 20%, and 30% attribution pixels

Statistics:
    - Overall summaries
    - Subgroups by prediction outcome, lesion group, and ROI size
    - 5,000 paired stratified bootstrap replicates
    - 95% bootstrap confidence intervals
    - Paired bootstrap differences between methods

Important:
    Integrated Gradients is expected to have been regenerated using
    N_STEPS = 200 before this script is run.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path.cwd()

TEST_PREDICTIONS_FILE = Path(
    "results/resnet50v2_finetuned/evaluation/"
    "test_predictions.csv"
)

ROI_XAI_MANIFEST_FILE = Path(
    "data/processed/cbis_ddsm/manifests/"
    "cbis_ddsm_roi_xai_manifest.csv"
)

GRADCAM_METRICS_FILE = Path(
    "results/resnet50v2_finetuned/"
    "gradcam_roi_alignment/"
    "gradcam_roi_metrics_per_image.csv"
)

GRADCAMPP_MANIFEST_FILE = Path(
    "results/resnet50v2_finetuned/"
    "gradcampp/"
    "gradcampp_test_manifest.csv"
)

INTEGRATED_GRADIENTS_MANIFEST_FILE = Path(
    "results/resnet50v2_finetuned/"
    "integrated_gradients/"
    "integrated_gradients_test_manifest.csv"
)

OUTPUT_DIR = Path(
    "results/resnet50v2_finetuned/"
    "xai_method_comparison"
)

PER_IMAGE_OUTPUT = (
    OUTPUT_DIR
    / "xai_methods_metrics_per_image.csv"
)

OVERALL_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "xai_methods_overall_summary.csv"
)

SUBGROUP_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "xai_methods_subgroup_summary.csv"
)

BOOTSTRAP_INTERVALS_OUTPUT = (
    OUTPUT_DIR
    / "xai_methods_bootstrap_intervals.csv"
)

BOOTSTRAP_DIFFERENCES_OUTPUT = (
    OUTPUT_DIR
    / "xai_methods_bootstrap_pairwise_differences.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "xai_methods_comparison_summary.json"
)

LATEX_TABLE_OUTPUT = (
    OUTPUT_DIR
    / "xai_methods_comparison_table.tex"
)

FIGURE_POINTING = (
    OUTPUT_DIR
    / "xai_pointing_game_comparison.png"
)

FIGURE_ENERGY = (
    OUTPUT_DIR
    / "xai_energy_inside_roi_boxplot.png"
)

FIGURE_RATIO = (
    OUTPUT_DIR
    / "xai_activation_ratio_boxplot.png"
)

FIGURE_IOU = (
    OUTPUT_DIR
    / "xai_iou_top10_boxplot.png"
)

FIGURE_SUBGROUP = (
    OUTPUT_DIR
    / "xai_pointing_game_by_outcome.png"
)

EXPECTED_TEST_CASES = 414
EXPECTED_IG_STEPS = 200

TOP_FRACTIONS = [
    0.10,
    0.20,
    0.30,
]

N_BOOTSTRAP = 5000
RANDOM_SEED = 42

EPSILON = 1e-10


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


def find_column(
    frame,
    candidates,
    required=True,
):
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    if required:
        raise ValueError(
            "Could not find any of these columns: "
            + ", ".join(candidates)
        )

    return None


def pathology_to_int(value):
    text = str(value).strip().upper()

    if text == "MALIGNANT":
        return 1

    if text == "BENIGN":
        return 0

    if text in {
        "0",
        "1",
    }:
        return int(text)

    raise ValueError(
        f"Unexpected pathology label: {value}"
    )


def classify_outcome(
    label,
    probability,
    threshold=0.5,
):
    prediction = int(
        probability >= threshold
    )

    if label == 1 and prediction == 1:
        return "TP"

    if label == 0 and prediction == 0:
        return "TN"

    if label == 0 and prediction == 1:
        return "FP"

    return "FN"


def load_binary_mask(path):
    image = Image.open(
        path
    ).convert(
        "L"
    )

    array = np.asarray(
        image
    )

    return (
        array > 0
    )


def load_heatmap(path):
    array = np.load(
        path
    ).astype(
        np.float32
    )

    if array.ndim != 2:
        raise ValueError(
            f"Expected 2-D heatmap, found shape "
            f"{array.shape} in {path}"
        )

    if array.shape != (
        512,
        512,
    ):
        raise ValueError(
            f"Expected 512 x 512 heatmap, found "
            f"{array.shape} in {path}"
        )

    if np.isnan(
        array
    ).any():
        raise ValueError(
            f"NaN values in heatmap: {path}"
        )

    if np.isinf(
        array
    ).any():
        raise ValueError(
            f"Infinite values in heatmap: {path}"
        )

    array = np.clip(
        array,
        0.0,
        None,
    )

    maximum = float(
        array.max()
    )

    if maximum > EPSILON:
        array = array / maximum

    return array


def union_roi_masks(
    roi_rows
):
    union = np.zeros(
        (
            512,
            512,
        ),
        dtype=bool,
    )

    usable = 0

    for _, row in roi_rows.iterrows():
        mask_path = resolve_path(
            row[
                "roi_mask_512_path"
            ]
        )

        if (
            mask_path is None
            or not mask_path.exists()
        ):
            continue

        mask = load_binary_mask(
            mask_path
        )

        if mask.shape != (
            512,
            512,
        ):
            raise ValueError(
                f"ROI mask has unexpected shape "
                f"{mask.shape}: {mask_path}"
            )

        union |= mask
        usable += 1

    if usable == 0:
        raise ValueError(
            "No usable ROI masks for mammogram."
        )

    if not union.any():
        raise ValueError(
            "Union ROI mask is empty."
        )

    return union


def evaluate_heatmap_against_roi(
    heatmap,
    roi_mask,
):
    roi_mask = roi_mask.astype(
        bool
    )

    outside_mask = ~roi_mask

    flat_max_index = int(
        np.argmax(
            heatmap
        )
    )

    max_row, max_col = np.unravel_index(
        flat_max_index,
        heatmap.shape,
    )

    pointing_hit = int(
        roi_mask[
            max_row,
            max_col
        ]
    )

    total_energy = float(
        np.sum(
            heatmap
        )
    )

    roi_energy = float(
        np.sum(
            heatmap[
                roi_mask
            ]
        )
    )

    energy_inside_roi = (
        roi_energy / total_energy
        if total_energy > EPSILON
        else 0.0
    )

    mean_inside = float(
        np.mean(
            heatmap[
                roi_mask
            ]
        )
    )

    mean_outside = float(
        np.mean(
            heatmap[
                outside_mask
            ]
        )
    )

    activation_ratio = (
        mean_inside
        /
        max(
            mean_outside,
            EPSILON,
        )
    )

    metrics = {
        "pointing_game_hit": pointing_hit,
        "energy_inside_roi": (
            energy_inside_roi
        ),
        "inside_outside_activation_ratio": (
            activation_ratio
        ),
    }

    flattened = heatmap.ravel()

    for fraction in TOP_FRACTIONS:
        percentile = (
            100.0
            * (
                1.0
                - fraction
            )
        )

        threshold = float(
            np.percentile(
                flattened,
                percentile,
            )
        )

        attribution_mask = (
            heatmap >= threshold
        )

        intersection = int(
            np.logical_and(
                attribution_mask,
                roi_mask,
            ).sum()
        )

        union = int(
            np.logical_or(
                attribution_mask,
                roi_mask,
            ).sum()
        )

        iou = (
            intersection / union
            if union > 0
            else 0.0
        )

        percentage = int(
            round(
                fraction
                * 100
            )
        )

        metrics[
            f"iou_top_{percentage}"
        ] = float(
            iou
        )

    return metrics


def metric_summary(
    frame,
    method,
):
    return {
        "method": method,
        "n": int(
            len(
                frame
            )
        ),
        "pointing_game_accuracy": float(
            frame[
                "pointing_game_hit"
            ].mean()
        ),
        "mean_energy_inside_roi": float(
            frame[
                "energy_inside_roi"
            ].mean()
        ),
        "median_energy_inside_roi": float(
            frame[
                "energy_inside_roi"
            ].median()
        ),
        "mean_activation_ratio": float(
            frame[
                "inside_outside_activation_ratio"
            ].mean()
        ),
        "median_activation_ratio": float(
            frame[
                "inside_outside_activation_ratio"
            ].median()
        ),
        "mean_iou_top_10": float(
            frame[
                "iou_top_10"
            ].mean()
        ),
        "median_iou_top_10": float(
            frame[
                "iou_top_10"
            ].median()
        ),
        "mean_iou_top_20": float(
            frame[
                "iou_top_20"
            ].mean()
        ),
        "median_iou_top_20": float(
            frame[
                "iou_top_20"
            ].median()
        ),
        "mean_iou_top_30": float(
            frame[
                "iou_top_30"
            ].mean()
        ),
        "median_iou_top_30": float(
            frame[
                "iou_top_30"
            ].median()
        ),
    }


def percentile_interval(values):
    return (
        float(
            np.percentile(
                values,
                2.5,
            )
        ),
        float(
            np.percentile(
                values,
                97.5,
            )
        ),
    )


def bootstrap_pvalue(
    differences
):
    values = np.asarray(
        differences,
        dtype=float,
    )

    p_lower = (
        np.sum(
            values <= 0
        )
        + 1
    ) / (
        len(values)
        + 1
    )

    p_upper = (
        np.sum(
            values >= 0
        )
        + 1
    ) / (
        len(values)
        + 1
    )

    return float(
        min(
            1.0,
            2.0
            * min(
                p_lower,
                p_upper,
            )
        )
    )


# ============================================================
# Validate files
# ============================================================

for path in [
    TEST_PREDICTIONS_FILE,
    ROI_XAI_MANIFEST_FILE,
    GRADCAM_METRICS_FILE,
    GRADCAMPP_MANIFEST_FILE,
    INTEGRATED_GRADIENTS_MANIFEST_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{path.resolve()}"
        )


# ============================================================
# Load predictions
# ============================================================

predictions_df = pd.read_csv(
    TEST_PREDICTIONS_FILE
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

predictions = predictions_df[
    [
        prediction_id_column,
        prediction_patient_column,
        prediction_pathology_column,
        prediction_probability_column,
    ]
].copy()

predictions.columns = [
    "full_image_id",
    "patient_id",
    "binary_pathology",
    "predicted_probability",
]

predictions[
    "full_image_id"
] = predictions[
    "full_image_id"
].astype(str)

predictions[
    "label"
] = predictions[
    "binary_pathology"
].apply(
    pathology_to_int
)

predictions[
    "outcome"
] = [
    classify_outcome(
        label,
        probability,
    )
    for label, probability in zip(
        predictions[
            "label"
        ],
        predictions[
            "predicted_probability"
        ],
    )
]


# ============================================================
# Load ROI manifest and build one union mask per mammogram
# ============================================================

roi_df = pd.read_csv(
    ROI_XAI_MANIFEST_FILE
)

roi_id_column = find_column(
    roi_df,
    [
        "full_image_id",
        "image_id",
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

roi_mask_column = find_column(
    roi_df,
    [
        "roi_mask_512_path",
    ],
)

lesion_type_column = find_column(
    roi_df,
    [
        "abnormality type",
        "abnormality_type",
    ],
)

roi_test = roi_df[
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

roi_test = roi_test.rename(
    columns={
        roi_id_column: "full_image_id",
        roi_mask_column: "roi_mask_512_path",
        lesion_type_column: "lesion_type",
    }
)

roi_test[
    "full_image_id"
] = roi_test[
    "full_image_id"
].astype(str)

roi_groups = {
    full_image_id: group.copy()
    for full_image_id, group in roi_test.groupby(
        "full_image_id"
    )
}

roi_case_rows = []

roi_masks = {}

for full_image_id, group in roi_groups.items():
    mask = union_roi_masks(
        group
    )

    roi_masks[
        full_image_id
    ] = mask

    lesion_types = sorted(
        set(
            group[
                "lesion_type"
            ]
            .dropna()
            .astype(str)
            .str.lower()
        )
    )

    if len(
        lesion_types
    ) == 1:
        lesion_group = lesion_types[
            0
        ]
    else:
        lesion_group = "mixed"

    roi_fraction = float(
        mask.mean()
    )

    roi_case_rows.append({
        "full_image_id": full_image_id,
        "lesion_group": lesion_group,
        "roi_foreground_fraction": (
            roi_fraction
        ),
        "n_roi_masks": int(
            len(
                group
            )
        ),
    })

roi_case_df = pd.DataFrame(
    roi_case_rows
)

if len(
    roi_case_df
) != EXPECTED_TEST_CASES:
    raise ValueError(
        "Expected "
        f"{EXPECTED_TEST_CASES} ROI-evaluable test mammograms, "
        f"but found {len(roi_case_df)}."
    )

q25 = float(
    roi_case_df[
        "roi_foreground_fraction"
    ].quantile(
        0.25
    )
)

q75 = float(
    roi_case_df[
        "roi_foreground_fraction"
    ].quantile(
        0.75
    )
)

roi_case_df[
    "roi_size_group"
] = np.where(
    roi_case_df[
        "roi_foreground_fraction"
    ]
    <= q25,
    "small",
    np.where(
        roi_case_df[
            "roi_foreground_fraction"
        ]
        >= q75,
        "large",
        "medium",
    ),
)


# ============================================================
# Grad-CAM baseline metrics
# ============================================================

gradcam_raw = pd.read_csv(
    GRADCAM_METRICS_FILE
)

gradcam_id_column = find_column(
    gradcam_raw,
    [
        "full_image_id",
        "image_id",
    ],
)

gradcam_column_map = {
    "pointing_game_hit": [
        "pointing_game_hit",
    ],
    "energy_inside_roi": [
        "energy_inside_roi",
    ],
    "inside_outside_activation_ratio": [
        "inside_outside_ratio",
    ],
    "iou_top_10": [
        "iou_top_10_percent",
    ],
    "iou_top_20": [
        "iou_top_20_percent",
    ],
    "iou_top_30": [
        "iou_top_30_percent",
    ],
}

gradcam = pd.DataFrame({
    "full_image_id": gradcam_raw[
        gradcam_id_column
    ].astype(str)
})

for canonical, candidates in gradcam_column_map.items():
    source = find_column(
        gradcam_raw,
        candidates,
    )

    gradcam[
        canonical
    ] = pd.to_numeric(
        gradcam_raw[
            source
        ],
        errors="coerce",
    )

if gradcam.isna().any().any():
    raise ValueError(
        "Missing/non-numeric values found in Grad-CAM metric file."
    )

gradcam[
    "method"
] = "Grad-CAM"


# ============================================================
# Grad-CAM++ raw heatmap evaluation
# ============================================================

gradcampp_manifest = pd.read_csv(
    GRADCAMPP_MANIFEST_FILE
)

gradcampp_id_column = find_column(
    gradcampp_manifest,
    [
        "full_image_id",
    ],
)

gradcampp_heatmap_column = find_column(
    gradcampp_manifest,
    [
        "heatmap_npy_path",
    ],
)

gradcampp_records = []

for row in gradcampp_manifest.itertuples(
    index=False
):
    full_image_id = str(
        getattr(
            row,
            gradcampp_id_column,
        )
    )

    if full_image_id not in roi_masks:
        continue

    heatmap_path = resolve_path(
        getattr(
            row,
            gradcampp_heatmap_column,
        )
    )

    if (
        heatmap_path is None
        or not heatmap_path.exists()
    ):
        raise FileNotFoundError(
            f"Grad-CAM++ heatmap missing: "
            f"{heatmap_path}"
        )

    metrics = evaluate_heatmap_against_roi(
        load_heatmap(
            heatmap_path
        ),
        roi_masks[
            full_image_id
        ],
    )

    gradcampp_records.append({
        "full_image_id": full_image_id,
        "method": "Grad-CAM++",
        **metrics,
    })

gradcampp = pd.DataFrame(
    gradcampp_records
)


# ============================================================
# Integrated Gradients raw heatmap evaluation
# ============================================================

ig_manifest = pd.read_csv(
    INTEGRATED_GRADIENTS_MANIFEST_FILE
)

ig_id_column = find_column(
    ig_manifest,
    [
        "full_image_id",
    ],
)

ig_heatmap_column = find_column(
    ig_manifest,
    [
        "heatmap_npy_path",
    ],
)

if "integration_steps" not in ig_manifest.columns:
    raise ValueError(
        "Integrated Gradients manifest does not contain "
        "'integration_steps'. Regenerate the 200-step maps first."
    )

ig_step_values = sorted(
    set(
        pd.to_numeric(
            ig_manifest[
                "integration_steps"
            ],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )
)

if ig_step_values != [
    EXPECTED_IG_STEPS
]:
    raise ValueError(
        "Integrated Gradients maps are not the required "
        f"{EXPECTED_IG_STEPS}-step maps. Found step values: "
        f"{ig_step_values}. Regenerate Integrated Gradients "
        "with N_STEPS = 200 before running this comparison."
    )

ig_records = []

for row in ig_manifest.itertuples(
    index=False
):
    full_image_id = str(
        getattr(
            row,
            ig_id_column,
        )
    )

    if full_image_id not in roi_masks:
        continue

    heatmap_path = resolve_path(
        getattr(
            row,
            ig_heatmap_column,
        )
    )

    if (
        heatmap_path is None
        or not heatmap_path.exists()
    ):
        raise FileNotFoundError(
            f"Integrated Gradients heatmap missing: "
            f"{heatmap_path}"
        )

    metrics = evaluate_heatmap_against_roi(
        load_heatmap(
            heatmap_path
        ),
        roi_masks[
            full_image_id
        ],
    )

    ig_records.append({
        "full_image_id": full_image_id,
        "method": "Integrated Gradients",
        **metrics,
    })

integrated_gradients = pd.DataFrame(
    ig_records
)


# ============================================================
# QA case counts
# ============================================================

for method_name, frame in [
    (
        "Grad-CAM",
        gradcam,
    ),
    (
        "Grad-CAM++",
        gradcampp,
    ),
    (
        "Integrated Gradients",
        integrated_gradients,
    ),
]:
    if len(
        frame
    ) != EXPECTED_TEST_CASES:
        raise ValueError(
            f"{method_name}: expected "
            f"{EXPECTED_TEST_CASES} cases, found {len(frame)}."
        )

    if frame[
        "full_image_id"
    ].duplicated().any():
        raise ValueError(
            f"{method_name}: duplicate full_image_id rows found."
        )


# ============================================================
# Combine all methods with shared metadata
# ============================================================

metadata = (
    predictions.merge(
        roi_case_df,
        on="full_image_id",
        how="inner",
        validate="one_to_one",
    )
)

if len(
    metadata
) != EXPECTED_TEST_CASES:
    raise ValueError(
        "Shared metadata cohort is not 414 cases."
    )

method_frames = []

for frame in [
    gradcam,
    gradcampp,
    integrated_gradients,
]:
    merged = metadata.merge(
        frame,
        on="full_image_id",
        how="inner",
        validate="one_to_one",
    )

    method_frames.append(
        merged
    )

all_metrics = pd.concat(
    method_frames,
    ignore_index=True,
)

method_order = [
    "Grad-CAM",
    "Grad-CAM++",
    "Integrated Gradients",
]

all_metrics[
    "method"
] = pd.Categorical(
    all_metrics[
        "method"
    ],
    categories=method_order,
    ordered=True,
)

all_metrics = all_metrics.sort_values(
    [
        "full_image_id",
        "method",
    ]
).reset_index(
    drop=True
)


# ============================================================
# Overall summary
# ============================================================

overall_rows = []

for method in method_order:
    subset = all_metrics[
        all_metrics[
            "method"
        ] == method
    ]

    overall_rows.append(
        metric_summary(
            subset,
            method,
        )
    )

overall_df = pd.DataFrame(
    overall_rows
)


# ============================================================
# Subgroup summary
# ============================================================

subgroup_rows = []

subgroup_definitions = [
    (
        "outcome",
        [
            "TP",
            "TN",
            "FP",
            "FN",
        ],
    ),
    (
        "lesion_group",
        sorted(
            all_metrics[
                "lesion_group"
            ].dropna().unique()
        ),
    ),
    (
        "roi_size_group",
        [
            "small",
            "medium",
            "large",
        ],
    ),
]

for group_type, groups in subgroup_definitions:
    for group in groups:
        for method in method_order:
            subset = all_metrics[
                (
                    all_metrics[
                        group_type
                    ] == group
                )
                &
                (
                    all_metrics[
                        "method"
                    ] == method
                )
            ]

            if len(
                subset
            ) == 0:
                continue

            row = metric_summary(
                subset,
                method,
            )

            row[
                "group_type"
            ] = group_type

            row[
                "group"
            ] = group

            subgroup_rows.append(
                row
            )

subgroup_df = pd.DataFrame(
    subgroup_rows
)


# ============================================================
# Paired bootstrap
# ============================================================

wide = all_metrics.pivot(
    index="full_image_id",
    columns="method",
    values=[
        "pointing_game_hit",
        "energy_inside_roi",
        "inside_outside_activation_ratio",
        "iou_top_10",
        "iou_top_20",
        "iou_top_30",
    ],
)

wide = wide.sort_index()

labels_by_id = (
    metadata
    .set_index(
        "full_image_id"
    )
    .loc[
        wide.index,
        "label",
    ]
    .to_numpy(
        dtype=int
    )
)

negative_indices = np.flatnonzero(
    labels_by_id == 0
)

positive_indices = np.flatnonzero(
    labels_by_id == 1
)

rng = np.random.default_rng(
    RANDOM_SEED
)

bootstrap_metric_functions = {
    "pointing_game_accuracy": (
        "pointing_game_hit",
        np.mean,
    ),
    "median_energy_inside_roi": (
        "energy_inside_roi",
        np.median,
    ),
    "median_activation_ratio": (
        "inside_outside_activation_ratio",
        np.median,
    ),
    "median_iou_top_10": (
        "iou_top_10",
        np.median,
    ),
}

bootstrap_values = {
    method: {
        metric: []
        for metric in bootstrap_metric_functions
    }
    for method in method_order
}

pairwise_pairs = [
    (
        "Grad-CAM++",
        "Grad-CAM",
    ),
    (
        "Integrated Gradients",
        "Grad-CAM",
    ),
    (
        "Integrated Gradients",
        "Grad-CAM++",
    ),
]

difference_values = {
    pair: {
        metric: []
        for metric in bootstrap_metric_functions
    }
    for pair in pairwise_pairs
}

print("=" * 72)
print("FINAL RQ2 XAI METHOD COMPARISON")
print("=" * 72)

print(
    f"\nMatched ROI-evaluable test mammograms: "
    f"{EXPECTED_TEST_CASES}"
)

print(
    "\nMethods:"
)

for method in method_order:
    print(
        f"  {method}"
    )

print(
    f"\nRunning "
    f"{N_BOOTSTRAP:,} paired stratified bootstrap replicates..."
)

for _ in range(
    N_BOOTSTRAP
):
    sampled_negative = rng.choice(
        negative_indices,
        size=len(
            negative_indices
        ),
        replace=True,
    )

    sampled_positive = rng.choice(
        positive_indices,
        size=len(
            positive_indices
        ),
        replace=True,
    )

    indices = np.concatenate(
        [
            sampled_negative,
            sampled_positive,
        ]
    )

    replicate = {}

    for method in method_order:
        replicate[
            method
        ] = {}

        for metric_name, (
            source_metric,
            function,
        ) in bootstrap_metric_functions.items():

            values = (
                wide[
                    source_metric
                ][
                    method
                ]
                .to_numpy(
                    dtype=float
                )[
                    indices
                ]
            )

            statistic = float(
                function(
                    values
                )
            )

            bootstrap_values[
                method
            ][
                metric_name
            ].append(
                statistic
            )

            replicate[
                method
            ][
                metric_name
            ] = statistic

    for method_a, method_b in pairwise_pairs:
        for metric_name in bootstrap_metric_functions:
            difference_values[
                (
                    method_a,
                    method_b,
                )
            ][
                metric_name
            ].append(
                replicate[
                    method_a
                ][
                    metric_name
                ]
                -
                replicate[
                    method_b
                ][
                    metric_name
                ]
            )


# ============================================================
# Bootstrap intervals
# ============================================================

interval_rows = []

for method in method_order:
    point_row = overall_df[
        overall_df[
            "method"
        ] == method
    ].iloc[
        0
    ]

    point_value_mapping = {
        "pointing_game_accuracy": (
            "pointing_game_accuracy"
        ),
        "median_energy_inside_roi": (
            "median_energy_inside_roi"
        ),
        "median_activation_ratio": (
            "median_activation_ratio"
        ),
        "median_iou_top_10": (
            "median_iou_top_10"
        ),
    }

    for metric_name, point_column in point_value_mapping.items():
        low, high = percentile_interval(
            bootstrap_values[
                method
            ][
                metric_name
            ]
        )

        interval_rows.append({
            "method": method,
            "metric": metric_name,
            "point_estimate": float(
                point_row[
                    point_column
                ]
            ),
            "ci_low": low,
            "ci_high": high,
        })

interval_df = pd.DataFrame(
    interval_rows
)


# ============================================================
# Paired bootstrap differences
# ============================================================

difference_rows = []

overall_lookup = (
    overall_df
    .set_index(
        "method"
    )
)

point_column_mapping = {
    "pointing_game_accuracy": (
        "pointing_game_accuracy"
    ),
    "median_energy_inside_roi": (
        "median_energy_inside_roi"
    ),
    "median_activation_ratio": (
        "median_activation_ratio"
    ),
    "median_iou_top_10": (
        "median_iou_top_10"
    ),
}

for method_a, method_b in pairwise_pairs:
    for metric_name, point_column in point_column_mapping.items():

        point_difference = (
            float(
                overall_lookup.loc[
                    method_a,
                    point_column,
                ]
            )
            -
            float(
                overall_lookup.loc[
                    method_b,
                    point_column,
                ]
            )
        )

        samples = difference_values[
            (
                method_a,
                method_b,
            )
        ][
            metric_name
        ]

        low, high = percentile_interval(
            samples
        )

        difference_rows.append({
            "comparison": (
                f"{method_a} - {method_b}"
            ),
            "method_a": method_a,
            "method_b": method_b,
            "metric": metric_name,
            "difference": (
                point_difference
            ),
            "ci_low": low,
            "ci_high": high,
            "bootstrap_p": (
                bootstrap_pvalue(
                    samples
                )
            ),
        })

difference_df = pd.DataFrame(
    difference_rows
)


# ============================================================
# Save outputs
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

all_metrics.to_csv(
    PER_IMAGE_OUTPUT,
    index=False,
)

overall_df.to_csv(
    OVERALL_SUMMARY_OUTPUT,
    index=False,
)

subgroup_df.to_csv(
    SUBGROUP_SUMMARY_OUTPUT,
    index=False,
)

interval_df.to_csv(
    BOOTSTRAP_INTERVALS_OUTPUT,
    index=False,
)

difference_df.to_csv(
    BOOTSTRAP_DIFFERENCES_OUTPUT,
    index=False,
)


# ============================================================
# Publication-ready LaTeX table
# ============================================================

latex_rows = []

for _, row in overall_df.iterrows():
    latex_rows.append(
        (
            f"{row['method']} & "
            f"{100.0 * row['pointing_game_accuracy']:.1f} & "
            f"{100.0 * row['median_energy_inside_roi']:.3f} & "
            f"{row['median_activation_ratio']:.3f} & "
            f"{100.0 * row['median_iou_top_10']:.3f} \\\\"
        )
    )

latex_text = r"""\begin{table}[htbp]
    \centering
    \caption{Localisation performance of Grad-CAM, Grad-CAM++ and
    Integrated Gradients on the matched 414-mammogram test cohort.}
    \label{tab:rq2_xai_method_comparison}
    \begin{tabular}{lrrrr}
        \toprule
        Method &
        Pointing-game (\%) &
        Median ROI energy (\%) &
        Median activation ratio &
        Median IoU top 10\% (\%) \\
        \midrule
""" + "\n".join(
    "        " + row
    for row in latex_rows
) + r"""
        \bottomrule
    \end{tabular}
\end{table}
"""

LATEX_TABLE_OUTPUT.write_text(
    latex_text,
    encoding="utf-8",
)


# ============================================================
# Figures
# ============================================================

# Pointing-game overall
fig = plt.figure(
    figsize=(
        7,
        5.5,
    )
)

values = [
    float(
        overall_df.loc[
            overall_df[
                "method"
            ] == method,
            "pointing_game_accuracy",
        ].iloc[
            0
        ]
    )
    for method in method_order
]

bars = plt.bar(
    method_order,
    values,
)

for bar, value in zip(
    bars,
    values,
):
    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.003,
        f"{100.0 * value:.1f}%",
        ha="center",
        va="bottom",
    )

plt.ylabel(
    "Pointing-game accuracy"
)

plt.title(
    "RQ2 Explainability Localisation Comparison"
)

plt.tight_layout()

fig.savefig(
    FIGURE_POINTING,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


def boxplot_metric(
    metric,
    ylabel,
    title,
    output_path,
):
    data = [
        all_metrics.loc[
            all_metrics[
                "method"
            ] == method,
            metric,
        ].to_numpy(
            dtype=float
        )
        for method in method_order
    ]

    fig = plt.figure(
        figsize=(
            8,
            6,
        )
    )

    plt.boxplot(
        data,
        tick_labels=method_order,
        showfliers=False,
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


boxplot_metric(
    "energy_inside_roi",
    "Energy fraction inside ROI",
    "Energy Inside Annotated Lesion ROI",
    FIGURE_ENERGY,
)

boxplot_metric(
    "inside_outside_activation_ratio",
    "Inside/outside activation ratio",
    "Attribution Concentration Inside vs Outside ROI",
    FIGURE_RATIO,
)

boxplot_metric(
    "iou_top_10",
    "IoU",
    "IoU of Top 10% Attribution Pixels with ROI",
    FIGURE_IOU,
)


# Pointing game by outcome
outcomes = [
    "TP",
    "TN",
    "FP",
    "FN",
]

x = np.arange(
    len(
        outcomes
    )
)

width = 0.24

fig = plt.figure(
    figsize=(
        10,
        6,
    )
)

for method_index, method in enumerate(
    method_order
):
    values = []

    for outcome in outcomes:
        subset = all_metrics[
            (
                all_metrics[
                    "method"
                ] == method
            )
            &
            (
                all_metrics[
                    "outcome"
                ] == outcome
            )
        ]

        values.append(
            float(
                subset[
                    "pointing_game_hit"
                ].mean()
            )
        )

    offset = (
        method_index - 1
    ) * width

    plt.bar(
        x + offset,
        values,
        width,
        label=method,
    )

plt.xticks(
    x,
    outcomes,
)

plt.ylabel(
    "Pointing-game accuracy"
)

plt.title(
    "Pointing-Game Accuracy by Prediction Outcome"
)

plt.legend()

plt.tight_layout()

fig.savefig(
    FIGURE_SUBGROUP,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# JSON summary
# ============================================================

summary = {
    "matched_test_mammograms": int(
        EXPECTED_TEST_CASES
    ),
    "methods": method_order,
    "integrated_gradients_steps": int(
        EXPECTED_IG_STEPS
    ),
    "roi_size_thresholds": {
        "q25": q25,
        "q75": q75,
    },
    "bootstrap_replicates": int(
        N_BOOTSTRAP
    ),
    "overall_summary": (
        overall_df.to_dict(
            orient="records"
        )
    ),
    "bootstrap_intervals": (
        interval_df.to_dict(
            orient="records"
        )
    ),
    "bootstrap_pairwise_differences": (
        difference_df.to_dict(
            orient="records"
        )
    ),
}

with SUMMARY_JSON.open(
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
print("RQ2 XAI METHOD COMPARISON COMPLETE")
print("=" * 72)

print(
    "\nOverall localisation summary:"
)

display_columns = [
    "method",
    "n",
    "pointing_game_accuracy",
    "median_energy_inside_roi",
    "median_activation_ratio",
    "median_iou_top_10",
    "median_iou_top_20",
    "median_iou_top_30",
]

print(
    overall_df[
        display_columns
    ].to_string(
        index=False
    )
)

print(
    "\nBootstrap 95% confidence intervals:"
)

print(
    interval_df.to_string(
        index=False
    )
)

print(
    "\nPaired bootstrap differences:"
)

print(
    difference_df.to_string(
        index=False
    )
)

print(
    "\nSubgroup counts:"
)

print(
    metadata[
        "outcome"
    ].value_counts()
)

print(
    metadata[
        "lesion_group"
    ].value_counts()
)

print(
    metadata[
        "roi_size_group"
    ].value_counts()
)

print(
    "\nSaved comparison directory:"
)

print(
    OUTPUT_DIR.resolve()
)