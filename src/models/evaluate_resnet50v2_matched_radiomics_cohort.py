from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path.cwd()

CNN_EVALUATION_DIR = Path(
    "results/resnet50v2_finetuned/evaluation"
)

CNN_VALIDATION_PREDICTIONS = (
    CNN_EVALUATION_DIR
    / "validation_predictions.csv"
)

CNN_TEST_PREDICTIONS = (
    CNN_EVALUATION_DIR
    / "test_predictions.csv"
)

RADIOMICS_EVALUATION_DIR = Path(
    "results/radiomics/radiomics_baseline"
)

RADIOMICS_VALIDATION_PREDICTIONS = (
    RADIOMICS_EVALUATION_DIR
    / "validation_predictions.csv"
)

RADIOMICS_TEST_PREDICTIONS = (
    RADIOMICS_EVALUATION_DIR
    / "test_predictions.csv"
)

OUTPUT_DIR = Path(
    "results/resnet50v2_finetuned/"
    "matched_radiomics_cohort"
)

DEFAULT_THRESHOLD = 0.5

VALIDATION_OUTPUT = (
    OUTPUT_DIR
    / "validation_predictions_matched.csv"
)

TEST_OUTPUT = (
    OUTPUT_DIR
    / "test_predictions_matched.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "matched_cohort_metrics_summary.json"
)

COMPARISON_CSV = (
    OUTPUT_DIR
    / "matched_cnn_vs_radiomics_test_metrics.csv"
)


# ============================================================
# Helpers
# ============================================================

def standardise_prediction_columns(
    frame,
    source_name
):
    """
    Standardise the saved prediction CSVs so that the matching
    logic is robust to the column naming used by earlier scripts.
    """

    output = frame.copy()

    id_candidates = [
        "image_id",
        "full_image_id",
    ]

    id_column = None

    for candidate in id_candidates:
        if candidate in output.columns:
            id_column = candidate
            break

    if id_column is None:
        raise ValueError(
            f"{source_name}: no image identifier column "
            f"found. Expected one of {id_candidates}."
        )

    probability_candidates = [
        "predicted_probability",
        "probability",
        "prediction",
    ]

    probability_column = None

    for candidate in probability_candidates:
        if candidate in output.columns:
            probability_column = candidate
            break

    if probability_column is None:
        raise ValueError(
            f"{source_name}: no probability column found. "
            f"Expected one of {probability_candidates}."
        )

    if "binary_pathology" not in output.columns:
        raise ValueError(
            f"{source_name}: binary_pathology column is missing."
        )

    if "patient_id" not in output.columns:
        raise ValueError(
            f"{source_name}: patient_id column is missing."
        )

    output = output.rename(
        columns={
            id_column: "full_image_id",
            probability_column: "predicted_probability",
        }
    )

    output[
        "predicted_probability"
    ] = pd.to_numeric(
        output[
            "predicted_probability"
        ],
        errors="coerce",
    )

    if output[
        "predicted_probability"
    ].isna().any():
        raise ValueError(
            f"{source_name}: non-numeric prediction probabilities found."
        )

    if output[
        "full_image_id"
    ].duplicated().any():
        duplicates = output.loc[
            output[
                "full_image_id"
            ].duplicated(
                keep=False
            ),
            [
                "full_image_id",
                "patient_id",
            ],
        ]

        raise ValueError(
            f"{source_name}: duplicate full_image_id rows found:\n"
            + duplicates.head(
                20
            ).to_string(
                index=False
            )
        )

    return output


def pathology_to_int(
    value
):
    text = str(
        value
    ).strip().upper()

    if text == "MALIGNANT":
        return 1

    if text == "BENIGN":
        return 0

    raise ValueError(
        f"Unexpected pathology label: {value}"
    )


def choose_threshold_max_f1(
    y_true,
    probabilities
):
    thresholds = np.unique(
        np.concatenate(
            [
                np.array(
                    [
                        0.0,
                        DEFAULT_THRESHOLD,
                        1.0,
                    ]
                ),
                probabilities,
            ]
        )
    )

    best_threshold = DEFAULT_THRESHOLD
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (
            probabilities
            >= threshold
        ).astype(
            int
        )

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = float(
                score
            )

            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        best_f1,
    )


def calculate_metrics(
    y_true,
    probabilities,
    threshold
):
    predictions = (
        probabilities
        >= threshold
    ).astype(
        int
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[
            0,
            1,
        ],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (
            tn + fp
        ) > 0
        else np.nan
    )

    return {
        "threshold": float(
            threshold
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall_sensitivity": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "specificity": float(
            specificity
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "tn": int(
            tn
        ),
        "fp": int(
            fp
        ),
        "fn": int(
            fn
        ),
        "tp": int(
            tp
        ),
    }


def print_metrics(
    title,
    metrics
):
    print(
        f"\n## {title}\n"
    )

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
        print(
            f"{key:20s}: "
            f"{metrics[key]:.4f}"
        )

    print(
        "confusion matrix: "
        f"TN={metrics['tn']}, "
        f"FP={metrics['fp']}, "
        f"FN={metrics['fn']}, "
        f"TP={metrics['tp']}"
    )


def merge_matched_cohort(
    cnn_frame,
    radiomics_frame,
    split_name
):
    cnn = standardise_prediction_columns(
        cnn_frame,
        f"CNN {split_name}",
    )

    radiomics = standardise_prediction_columns(
        radiomics_frame,
        f"Radiomics {split_name}",
    )

    matched = cnn.merge(
        radiomics[
            [
                "full_image_id",
                "patient_id",
                "binary_pathology",
                "predicted_probability",
            ]
        ],
        on="full_image_id",
        how="inner",
        suffixes=(
            "_cnn",
            "_radiomics",
        ),
        validate="one_to_one",
    )

    if len(
        matched
    ) != len(
        radiomics
    ):
        missing_count = (
            len(
                radiomics
            )
            - len(
                matched
            )
        )

        raise ValueError(
            f"{split_name}: {missing_count} radiomics rows "
            f"were not found in CNN predictions."
        )

    matched[
        "patient_id_match"
    ] = (
        matched[
            "patient_id_cnn"
        ].astype(
            str
        )
        ==
        matched[
            "patient_id_radiomics"
        ].astype(
            str
        )
    )

    matched[
        "pathology_match"
    ] = (
        matched[
            "binary_pathology_cnn"
        ].astype(
            str
        )
        ==
        matched[
            "binary_pathology_radiomics"
        ].astype(
            str
        )
    )

    if not matched[
        "patient_id_match"
    ].all():
        raise ValueError(
            f"{split_name}: patient ID mismatch found."
        )

    if not matched[
        "pathology_match"
    ].all():
        raise ValueError(
            f"{split_name}: pathology label mismatch found."
        )

    matched[
        "label"
    ] = matched[
        "binary_pathology_cnn"
    ].apply(
        pathology_to_int
    ).to_numpy(
        dtype=int
    )

    return matched


def save_roc_comparison(
    y_true,
    cnn_probabilities,
    radiomics_probabilities,
    output_path,
):
    cnn_fpr, cnn_tpr, _ = roc_curve(
        y_true,
        cnn_probabilities,
    )

    rad_fpr, rad_tpr, _ = roc_curve(
        y_true,
        radiomics_probabilities,
    )

    cnn_auc = roc_auc_score(
        y_true,
        cnn_probabilities,
    )

    rad_auc = roc_auc_score(
        y_true,
        radiomics_probabilities,
    )

    fig = plt.figure(
        figsize=(
            7,
            6,
        )
    )

    plt.plot(
        cnn_fpr,
        cnn_tpr,
        label=f"CNN AUC = {cnn_auc:.3f}",
    )

    plt.plot(
        rad_fpr,
        rad_tpr,
        label=f"Radiomics AUC = {rad_auc:.3f}",
    )

    plt.plot(
        [
            0,
            1,
        ],
        [
            0,
            1,
        ],
        linestyle="--",
        label="Chance",
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "Matched Test Cohort ROC Curves"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
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


def save_pr_comparison(
    y_true,
    cnn_probabilities,
    radiomics_probabilities,
    output_path,
):
    cnn_precision, cnn_recall, _ = (
        precision_recall_curve(
            y_true,
            cnn_probabilities,
        )
    )

    rad_precision, rad_recall, _ = (
        precision_recall_curve(
            y_true,
            radiomics_probabilities,
        )
    )

    cnn_ap = average_precision_score(
        y_true,
        cnn_probabilities,
    )

    rad_ap = average_precision_score(
        y_true,
        radiomics_probabilities,
    )

    fig = plt.figure(
        figsize=(
            7,
            6,
        )
    )

    plt.plot(
        cnn_recall,
        cnn_precision,
        label=f"CNN AP = {cnn_ap:.3f}",
    )

    plt.plot(
        rad_recall,
        rad_precision,
        label=f"Radiomics AP = {rad_ap:.3f}",
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Matched Test Cohort Precision-Recall Curves"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
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


# ============================================================
# Validate source files
# ============================================================

required_files = [
    CNN_VALIDATION_PREDICTIONS,
    CNN_TEST_PREDICTIONS,
    RADIOMICS_VALIDATION_PREDICTIONS,
    RADIOMICS_TEST_PREDICTIONS,
]

missing_files = [
    path
    for path in required_files
    if not path.exists()
]

if missing_files:
    raise FileNotFoundError(
        "Required prediction files are missing:\n"
        + "\n".join(
            str(
                path.resolve()
            )
            for path in missing_files
        )
    )


# ============================================================
# Load and match
# ============================================================

cnn_validation = pd.read_csv(
    CNN_VALIDATION_PREDICTIONS
)

cnn_test = pd.read_csv(
    CNN_TEST_PREDICTIONS
)

radiomics_validation = pd.read_csv(
    RADIOMICS_VALIDATION_PREDICTIONS
)

radiomics_test = pd.read_csv(
    RADIOMICS_TEST_PREDICTIONS
)

matched_validation = merge_matched_cohort(
    cnn_validation,
    radiomics_validation,
    "validation",
)

matched_test = merge_matched_cohort(
    cnn_test,
    radiomics_test,
    "test",
)

print("=" * 72)
print("MATCHED CNN / RADIOMICS COHORT EVALUATION")
print("=" * 72)

print(
    f"\nMatched validation mammograms: "
    f"{len(matched_validation):,}"
)

print(
    f"Matched test mammograms: "
    f"{len(matched_test):,}"
)

print(
    "\nValidation class counts:"
)

print(
    matched_validation[
        "binary_pathology_cnn"
    ].value_counts()
)

print(
    "\nTest class counts:"
)

print(
    matched_test[
        "binary_pathology_cnn"
    ].value_counts()
)


# ============================================================
# CNN threshold selection on matched validation only
# ============================================================

y_validation = matched_validation[
    "label"
].to_numpy(
    dtype=int
)

cnn_validation_probabilities = matched_validation[
    "predicted_probability_cnn"
].to_numpy(
    dtype=float
)

(
    cnn_selected_threshold,
    cnn_validation_best_f1,
) = choose_threshold_max_f1(
    y_validation,
    cnn_validation_probabilities,
)

print(
    "\nCNN threshold selection on matched validation cohort:"
)

print(
    f"Default threshold: "
    f"{DEFAULT_THRESHOLD:.3f}"
)

print(
    f"Validation-selected threshold "
    f"(max F1): "
    f"{cnn_selected_threshold:.3f}"
)

print(
    f"Validation F1 at selected threshold: "
    f"{cnn_validation_best_f1:.4f}"
)


# ============================================================
# Metrics
# ============================================================

y_test = matched_test[
    "label"
].to_numpy(
    dtype=int
)

cnn_test_probabilities = matched_test[
    "predicted_probability_cnn"
].to_numpy(
    dtype=float
)

radiomics_test_probabilities = matched_test[
    "predicted_probability_radiomics"
].to_numpy(
    dtype=float
)

cnn_validation_default = calculate_metrics(
    y_validation,
    cnn_validation_probabilities,
    DEFAULT_THRESHOLD,
)

cnn_validation_selected = calculate_metrics(
    y_validation,
    cnn_validation_probabilities,
    cnn_selected_threshold,
)

cnn_test_default = calculate_metrics(
    y_test,
    cnn_test_probabilities,
    DEFAULT_THRESHOLD,
)

cnn_test_selected = calculate_metrics(
    y_test,
    cnn_test_probabilities,
    cnn_selected_threshold,
)

radiomics_test_default = calculate_metrics(
    y_test,
    radiomics_test_probabilities,
    DEFAULT_THRESHOLD,
)

print_metrics(
    "Matched CNN validation metrics at threshold 0.5",
    cnn_validation_default,
)

print_metrics(
    "Matched CNN validation metrics at selected threshold",
    cnn_validation_selected,
)

print_metrics(
    "Matched CNN test metrics at threshold 0.5",
    cnn_test_default,
)

print_metrics(
    "Matched CNN test metrics at validation-selected threshold",
    cnn_test_selected,
)

print_metrics(
    "Matched radiomics test metrics at threshold 0.5",
    radiomics_test_default,
)


# ============================================================
# Save matched prediction tables
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

validation_output = matched_validation[
    [
        "full_image_id",
        "patient_id_cnn",
        "binary_pathology_cnn",
        "label",
        "predicted_probability_cnn",
        "predicted_probability_radiomics",
    ]
].copy()

validation_output = validation_output.rename(
    columns={
        "patient_id_cnn": "patient_id",
        "binary_pathology_cnn": "binary_pathology",
        "predicted_probability_cnn": "cnn_probability",
        "predicted_probability_radiomics": "radiomics_probability",
    }
)

validation_output[
    "cnn_predicted_label_default"
] = (
    validation_output[
        "cnn_probability"
    ]
    >= DEFAULT_THRESHOLD
).astype(
    int
)

validation_output[
    "cnn_predicted_label_selected"
] = (
    validation_output[
        "cnn_probability"
    ]
    >= cnn_selected_threshold
).astype(
    int
)

validation_output.to_csv(
    VALIDATION_OUTPUT,
    index=False,
)

test_output = matched_test[
    [
        "full_image_id",
        "patient_id_cnn",
        "binary_pathology_cnn",
        "label",
        "predicted_probability_cnn",
        "predicted_probability_radiomics",
    ]
].copy()

test_output = test_output.rename(
    columns={
        "patient_id_cnn": "patient_id",
        "binary_pathology_cnn": "binary_pathology",
        "predicted_probability_cnn": "cnn_probability",
        "predicted_probability_radiomics": "radiomics_probability",
    }
)

test_output[
    "cnn_predicted_label_default"
] = (
    test_output[
        "cnn_probability"
    ]
    >= DEFAULT_THRESHOLD
).astype(
    int
)

test_output[
    "cnn_predicted_label_selected"
] = (
    test_output[
        "cnn_probability"
    ]
    >= cnn_selected_threshold
).astype(
    int
)

test_output.to_csv(
    TEST_OUTPUT,
    index=False,
)


# ============================================================
# Comparison table
# ============================================================

comparison_rows = []

for model_name, metrics in [
    (
        "CNN-only matched",
        cnn_test_default,
    ),
    (
        "Radiomics-only matched",
        radiomics_test_default,
    ),
]:
    comparison_rows.append({
        "model": model_name,
        "test_n": int(
            len(
                matched_test
            )
        ),
        "threshold": float(
            metrics[
                "threshold"
            ]
        ),
        "roc_auc": float(
            metrics[
                "roc_auc"
            ]
        ),
        "average_precision": float(
            metrics[
                "average_precision"
            ]
        ),
        "accuracy": float(
            metrics[
                "accuracy"
            ]
        ),
        "precision": float(
            metrics[
                "precision"
            ]
        ),
        "recall_sensitivity": float(
            metrics[
                "recall_sensitivity"
            ]
        ),
        "specificity": float(
            metrics[
                "specificity"
            ]
        ),
        "f1": float(
            metrics[
                "f1"
            ]
        ),
    })

comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    COMPARISON_CSV,
    index=False,
)


# ============================================================
# Curves
# ============================================================

save_roc_comparison(
    y_test,
    cnn_test_probabilities,
    radiomics_test_probabilities,
    OUTPUT_DIR
    / "matched_test_roc_cnn_vs_radiomics.png",
)

save_pr_comparison(
    y_test,
    cnn_test_probabilities,
    radiomics_test_probabilities,
    OUTPUT_DIR
    / "matched_test_precision_recall_cnn_vs_radiomics.png",
)


# ============================================================
# Summary
# ============================================================

summary = {
    "matched_validation_n": int(
        len(
            matched_validation
        )
    ),
    "matched_test_n": int(
        len(
            matched_test
        )
    ),
    "cnn_threshold_selection": {
        "method": "matched validation maximum F1",
        "selected_threshold": float(
            cnn_selected_threshold
        ),
        "validation_f1": float(
            cnn_validation_best_f1
        ),
    },
    "cnn_matched_validation_threshold_0_5": (
        cnn_validation_default
    ),
    "cnn_matched_validation_selected_threshold": (
        cnn_validation_selected
    ),
    "cnn_matched_test_threshold_0_5": (
        cnn_test_default
    ),
    "cnn_matched_test_selected_threshold": (
        cnn_test_selected
    ),
    "radiomics_matched_test_threshold_0_5": (
        radiomics_test_default
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
# Final console summary
# ============================================================

print("\n")
print("=" * 72)
print("MATCHED COHORT EVALUATION COMPLETE")
print("=" * 72)

print(
    f"\nMatched validation cohort: "
    f"{len(matched_validation):,}"
)

print(
    f"Matched test cohort: "
    f"{len(matched_test):,}"
)

print(
    "\nMatched test ROC AUC:"
)

print(
    f"CNN-only       : "
    f"{cnn_test_default['roc_auc']:.4f}"
)

print(
    f"Radiomics-only : "
    f"{radiomics_test_default['roc_auc']:.4f}"
)

print(
    "\nMatched test average precision:"
)

print(
    f"CNN-only       : "
    f"{cnn_test_default['average_precision']:.4f}"
)

print(
    f"Radiomics-only : "
    f"{radiomics_test_default['average_precision']:.4f}"
)

print(
    "\nSaved matched comparison:"
)

print(
    COMPARISON_CSV.resolve()
)

print(
    "\nSaved matched prediction tables:"
)

print(
    VALIDATION_OUTPUT.resolve()
)

print(
    TEST_OUTPUT.resolve()
)

print(
    "\nSaved summary:"
)

print(
    SUMMARY_JSON.resolve()
)

print(
    "\nSaved figures:"
)

print(
    OUTPUT_DIR.resolve()
)
