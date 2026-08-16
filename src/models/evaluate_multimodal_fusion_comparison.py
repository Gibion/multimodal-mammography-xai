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

CNN_MATCHED_FILE = Path(
    "results/resnet50v2_finetuned/matched_radiomics_cohort/"
    "test_predictions_matched.csv"
)

RADIOMICS_RESULTS_FILE = Path(
    "results/radiomics/radiomics_baseline/"
    "test_predictions.csv"
)

FUSION_RESULTS_FILE = Path(
    "results/multimodal_fusion/logistic_regression/"
    "test_predictions.csv"
)

CNN_MATCHED_SUMMARY = Path(
    "results/resnet50v2_finetuned/matched_radiomics_cohort/"
    "matched_cohort_metrics_summary.json"
)

RADIOMICS_SUMMARY = Path(
    "results/radiomics/radiomics_baseline/"
    "metrics_summary.json"
)

FUSION_SUMMARY = Path(
    "results/multimodal_fusion/logistic_regression/"
    "fusion_logistic_regression_summary.json"
)

OUTPUT_DIR = Path(
    "results/multimodal_fusion/comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

COMBINED_PREDICTIONS_FILE = (
    OUTPUT_DIR / "rq1_combined_test_predictions.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR / "rq1_model_comparison_summary.csv"
)

BOOTSTRAP_MODEL_CSV = (
    OUTPUT_DIR / "rq1_bootstrap_model_intervals.csv"
)

BOOTSTRAP_DIFF_CSV = (
    OUTPUT_DIR / "rq1_bootstrap_pairwise_differences.csv"
)

LATEX_TABLE_FILE = (
    OUTPUT_DIR / "rq1_model_comparison_table.tex"
)

SUMMARY_JSON = (
    OUTPUT_DIR / "rq1_model_comparison_summary.json"
)

FIGURE_ROC = (
    OUTPUT_DIR / "rq1_roc_comparison.png"
)

FIGURE_PR = (
    OUTPUT_DIR / "rq1_precision_recall_comparison.png"
)

FIGURE_METRICS = (
    OUTPUT_DIR / "rq1_threshold_0_5_metric_comparison.png"
)

RANDOM_SEED = 42
N_BOOTSTRAP = 5000
CI_LOWER = 2.5
CI_UPPER = 97.5
DEFAULT_THRESHOLD = 0.5


# ============================================================
# Helpers
# ============================================================

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


def load_json_if_exists(path):
    if not path.exists():
        return {}

    with path.open(
        "r",
        encoding="utf-8"
    ) as handle:
        return json.load(handle)


def standardise_cnn_matched(frame):
    required = {
        "full_image_id",
        "patient_id",
        "binary_pathology",
        "cnn_probability",
        "radiomics_probability",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            "CNN matched prediction file is missing columns: "
            + ", ".join(sorted(missing))
        )

    output = frame[
        [
            "full_image_id",
            "patient_id",
            "binary_pathology",
            "cnn_probability",
            "radiomics_probability",
        ]
    ].copy()

    output = output.rename(
        columns={
            "cnn_probability": "cnn_probability",
            "radiomics_probability": "radiomics_probability_matched",
        }
    )

    return output


def standardise_single_model(
    frame,
    model_name,
):
    required = {
        "full_image_id",
        "patient_id",
        "binary_pathology",
        "predicted_probability",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"{model_name} prediction file is missing columns: "
            + ", ".join(sorted(missing))
        )

    output = frame[
        [
            "full_image_id",
            "patient_id",
            "binary_pathology",
            "predicted_probability",
        ]
    ].copy()

    output = output.rename(
        columns={
            "patient_id": f"patient_id_{model_name}",
            "binary_pathology": f"binary_pathology_{model_name}",
            "predicted_probability": f"{model_name}_probability",
        }
    )

    return output


def calculate_threshold_metrics(
    y_true,
    probabilities,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "threshold": float(threshold),
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
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def stratified_bootstrap_indices(
    y_true,
    rng,
):
    negative_indices = np.flatnonzero(
        y_true == 0
    )

    positive_indices = np.flatnonzero(
        y_true == 1
    )

    sampled_negative = rng.choice(
        negative_indices,
        size=len(negative_indices),
        replace=True,
    )

    sampled_positive = rng.choice(
        positive_indices,
        size=len(positive_indices),
        replace=True,
    )

    return np.concatenate(
        [
            sampled_negative,
            sampled_positive,
        ]
    )


def percentile_interval(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    return (
        float(
            np.percentile(
                values,
                CI_LOWER,
            )
        ),
        float(
            np.percentile(
                values,
                CI_UPPER,
            )
        ),
    )


def bootstrap_two_sided_pvalue(
    difference_samples,
):
    """
    Paired-bootstrap sign-based two-sided p-value.

    This is an empirical bootstrap comparison, not a DeLong test.
    """

    differences = np.asarray(
        difference_samples,
        dtype=float,
    )

    p_lower = (
        np.sum(
            differences <= 0
        )
        + 1
    ) / (
        len(differences)
        + 1
    )

    p_upper = (
        np.sum(
            differences >= 0
        )
        + 1
    ) / (
        len(differences)
        + 1
    )

    return float(
        min(
            1.0,
            2.0 * min(
                p_lower,
                p_upper,
            )
        )
    )


def get_selected_thresholds():
    cnn_summary = load_json_if_exists(
        CNN_MATCHED_SUMMARY
    )

    radiomics_summary = load_json_if_exists(
        RADIOMICS_SUMMARY
    )

    fusion_summary = load_json_if_exists(
        FUSION_SUMMARY
    )

    cnn_threshold = DEFAULT_THRESHOLD
    radiomics_threshold = DEFAULT_THRESHOLD
    fusion_threshold = DEFAULT_THRESHOLD

    try:
        cnn_threshold = float(
            cnn_summary[
                "cnn_threshold_selection"
            ][
                "selected_threshold"
            ]
        )
    except Exception:
        pass

    try:
        radiomics_threshold = float(
            radiomics_summary[
                "threshold_selection"
            ][
                "selected_threshold"
            ]
        )
    except Exception:
        pass

    try:
        fusion_threshold = float(
            fusion_summary[
                "validation_selected_threshold"
            ]
        )
    except Exception:
        pass

    return {
        "CNN-only": cnn_threshold,
        "Radiomics-only": radiomics_threshold,
        "Fusion": fusion_threshold,
    }


def save_roc_figure(
    y_true,
    model_probabilities,
):
    fig = plt.figure(
        figsize=(7.5, 6.5)
    )

    for model_name, probabilities in model_probabilities.items():
        fpr, tpr, _ = roc_curve(
            y_true,
            probabilities,
        )

        auc = roc_auc_score(
            y_true,
            probabilities,
        )

        plt.plot(
            fpr,
            tpr,
            label=(
                f"{model_name} "
                f"(AUC = {auc:.3f})"
            ),
        )

    plt.plot(
        [0, 1],
        [0, 1],
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
        "RQ1 Matched Test Cohort ROC Comparison"
    )
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    fig.savefig(
        FIGURE_ROC,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_pr_figure(
    y_true,
    model_probabilities,
):
    fig = plt.figure(
        figsize=(7.5, 6.5)
    )

    for model_name, probabilities in model_probabilities.items():
        precision, recall, _ = (
            precision_recall_curve(
                y_true,
                probabilities,
            )
        )

        ap = average_precision_score(
            y_true,
            probabilities,
        )

        plt.plot(
            recall,
            precision,
            label=(
                f"{model_name} "
                f"(AP = {ap:.3f})"
            ),
        )

    prevalence = float(
        np.mean(
            y_true
        )
    )

    plt.axhline(
        prevalence,
        linestyle="--",
        label=(
            "Malignant prevalence "
            f"({prevalence:.3f})"
        ),
    )

    plt.xlabel(
        "Recall"
    )
    plt.ylabel(
        "Precision"
    )
    plt.title(
        "RQ1 Matched Test Cohort Precision-Recall Comparison"
    )
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    fig.savefig(
        FIGURE_PR,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_threshold_metric_figure(
    summary_df,
):
    metrics = [
        "roc_auc",
        "average_precision",
        "accuracy",
        "recall_sensitivity",
        "specificity",
        "f1",
    ]

    labels = [
        "ROC AUC",
        "Average\nPrecision",
        "Accuracy",
        "Sensitivity",
        "Specificity",
        "F1",
    ]

    models = [
        "CNN-only",
        "Radiomics-only",
        "Fusion",
    ]

    x = np.arange(
        len(metrics)
    )

    width = 0.24

    fig = plt.figure(
        figsize=(11, 6.5)
    )

    for model_index, model_name in enumerate(models):
        row = summary_df[
            summary_df[
                "model"
            ] == model_name
        ].iloc[0]

        values = [
            float(
                row[
                    metric
                ]
            )
            for metric in metrics
        ]

        offset = (
            model_index - 1
        ) * width

        bars = plt.bar(
            x + offset,
            values,
            width,
            label=model_name,
        )

        for bar, value in zip(
            bars,
            values,
        ):
            plt.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 0.008,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    plt.xticks(
        x,
        labels,
    )

    plt.ylabel(
        "Performance"
    )

    plt.ylim(
        0,
        0.85,
    )

    plt.title(
        "RQ1 Model Performance on Matched Test Cohort "
        "(Threshold = 0.5)"
    )

    plt.legend()
    plt.grid(
        axis="y",
        alpha=0.25,
    )
    plt.tight_layout()

    fig.savefig(
        FIGURE_METRICS,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Load and align saved predictions
# ============================================================

for path in [
    CNN_MATCHED_FILE,
    RADIOMICS_RESULTS_FILE,
    FUSION_RESULTS_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required prediction file not found:\n"
            f"{path.resolve()}"
        )

cnn_matched = standardise_cnn_matched(
    pd.read_csv(
        CNN_MATCHED_FILE
    )
)

radiomics = standardise_single_model(
    pd.read_csv(
        RADIOMICS_RESULTS_FILE
    ),
    "radiomics",
)

fusion = standardise_single_model(
    pd.read_csv(
        FUSION_RESULTS_FILE
    ),
    "fusion",
)

combined = cnn_matched.merge(
    radiomics,
    on="full_image_id",
    how="inner",
    validate="one_to_one",
)

combined = combined.merge(
    fusion,
    on="full_image_id",
    how="inner",
    validate="one_to_one",
)

if len(
    combined
) != 414:
    raise ValueError(
        "Expected 414 matched test mammograms, "
        f"but found {len(combined)}."
    )

# Metadata agreement.
for model_suffix in [
    "radiomics",
    "fusion",
]:
    patient_match = (
        combined[
            "patient_id"
        ].astype(str)
        ==
        combined[
            f"patient_id_{model_suffix}"
        ].astype(str)
    )

    pathology_match = (
        combined[
            "binary_pathology"
        ].astype(str)
        ==
        combined[
            f"binary_pathology_{model_suffix}"
        ].astype(str)
    )

    if not patient_match.all():
        raise ValueError(
            f"Patient mismatch found for {model_suffix}."
        )

    if not pathology_match.all():
        raise ValueError(
            f"Pathology mismatch found for {model_suffix}."
        )

# Confirm the matched-radiomics probability and independently saved
# radiomics probability are the same to numerical precision.
max_radiomics_probability_difference = float(
    np.max(
        np.abs(
            combined[
                "radiomics_probability_matched"
            ].to_numpy(dtype=float)
            -
            combined[
                "radiomics_probability"
            ].to_numpy(dtype=float)
        )
    )
)

if (
    max_radiomics_probability_difference
    > 1e-8
):
    raise ValueError(
        "Radiomics predictions disagree between saved result files. "
        f"Maximum difference: "
        f"{max_radiomics_probability_difference:.12f}"
    )


# ============================================================
# Basic cohort information
# ============================================================

combined[
    "label"
] = combined[
    "binary_pathology"
].apply(
    pathology_to_int
).to_numpy(
    dtype=int
)

y_true = combined[
    "label"
].to_numpy(
    dtype=int
)

model_probabilities = {
    "CNN-only": combined[
        "cnn_probability"
    ].to_numpy(
        dtype=float
    ),
    "Radiomics-only": combined[
        "radiomics_probability"
    ].to_numpy(
        dtype=float
    ),
    "Fusion": combined[
        "fusion_probability"
    ].to_numpy(
        dtype=float
    ),
}

print("=" * 72)
print("FINAL RQ1 MULTIMODAL FUSION COMPARISON")
print("=" * 72)

print(
    f"\nMatched test mammograms: "
    f"{len(combined):,}"
)

print(
    "\nClass counts:"
)

print(
    combined[
        "binary_pathology"
    ].value_counts()
)

print(
    "\nMaximum duplicate radiomics probability difference:"
)

print(
    f"{max_radiomics_probability_difference:.12f}"
)


# ============================================================
# Threshold = 0.5 summary
# ============================================================

threshold_05_rows = []

metrics_05_by_model = {}

for model_name, probabilities in model_probabilities.items():
    metrics = calculate_threshold_metrics(
        y_true,
        probabilities,
        DEFAULT_THRESHOLD,
    )

    metrics_05_by_model[
        model_name
    ] = metrics

    threshold_05_rows.append({
        "model": model_name,
        "test_n": int(
            len(
                y_true
            )
        ),
        **metrics,
    })

summary_df = pd.DataFrame(
    threshold_05_rows
)

print(
    "\nThreshold 0.5 comparison:"
)

print(
    summary_df[
        [
            "model",
            "roc_auc",
            "average_precision",
            "accuracy",
            "precision",
            "recall_sensitivity",
            "specificity",
            "f1",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Model-specific validation-selected threshold summary
# ============================================================

selected_thresholds = get_selected_thresholds()

selected_threshold_rows = []

for model_name, probabilities in model_probabilities.items():
    threshold = selected_thresholds[
        model_name
    ]

    metrics = calculate_threshold_metrics(
        y_true,
        probabilities,
        threshold,
    )

    selected_threshold_rows.append({
        "model": model_name,
        "test_n": int(
            len(
                y_true
            )
        ),
        **metrics,
    })

selected_threshold_df = pd.DataFrame(
    selected_threshold_rows
)

print(
    "\nValidation-selected threshold comparison:"
)

print(
    selected_threshold_df[
        [
            "model",
            "threshold",
            "roc_auc",
            "average_precision",
            "accuracy",
            "precision",
            "recall_sensitivity",
            "specificity",
            "f1",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Paired stratified bootstrap
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

model_bootstrap = {
    model_name: {
        "roc_auc": [],
        "average_precision": [],
    }
    for model_name in model_probabilities
}

pairwise_pairs = [
    (
        "Fusion",
        "CNN-only",
    ),
    (
        "Fusion",
        "Radiomics-only",
    ),
    (
        "CNN-only",
        "Radiomics-only",
    ),
]

difference_bootstrap = {
    pair: {
        "roc_auc": [],
        "average_precision": [],
    }
    for pair in pairwise_pairs
}

print(
    f"\nRunning {N_BOOTSTRAP:,} paired stratified bootstrap replicates..."
)

for iteration in range(
    N_BOOTSTRAP
):
    indices = stratified_bootstrap_indices(
        y_true,
        rng,
    )

    y_boot = y_true[
        indices
    ]

    replicate_metrics = {}

    for model_name, probabilities in model_probabilities.items():
        p_boot = probabilities[
            indices
        ]

        auc = roc_auc_score(
            y_boot,
            p_boot,
        )

        ap = average_precision_score(
            y_boot,
            p_boot,
        )

        model_bootstrap[
            model_name
        ][
            "roc_auc"
        ].append(
            auc
        )

        model_bootstrap[
            model_name
        ][
            "average_precision"
        ].append(
            ap
        )

        replicate_metrics[
            model_name
        ] = {
            "roc_auc": auc,
            "average_precision": ap,
        }

    for model_a, model_b in pairwise_pairs:
        for metric in [
            "roc_auc",
            "average_precision",
        ]:
            difference_bootstrap[
                (
                    model_a,
                    model_b,
                )
            ][
                metric
            ].append(
                replicate_metrics[
                    model_a
                ][
                    metric
                ]
                -
                replicate_metrics[
                    model_b
                ][
                    metric
                ]
            )


# ============================================================
# Bootstrap model intervals
# ============================================================

model_interval_rows = []

for model_name, probabilities in model_probabilities.items():
    point_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    point_ap = average_precision_score(
        y_true,
        probabilities,
    )

    auc_low, auc_high = percentile_interval(
        model_bootstrap[
            model_name
        ][
            "roc_auc"
        ]
    )

    ap_low, ap_high = percentile_interval(
        model_bootstrap[
            model_name
        ][
            "average_precision"
        ]
    )

    model_interval_rows.append({
        "model": model_name,
        "test_n": int(
            len(
                y_true
            )
        ),
        "roc_auc": float(
            point_auc
        ),
        "roc_auc_ci_low": auc_low,
        "roc_auc_ci_high": auc_high,
        "average_precision": float(
            point_ap
        ),
        "average_precision_ci_low": ap_low,
        "average_precision_ci_high": ap_high,
    })

model_interval_df = pd.DataFrame(
    model_interval_rows
)


# ============================================================
# Bootstrap paired differences
# ============================================================

difference_rows = []

for model_a, model_b in pairwise_pairs:
    point_auc_difference = (
        metrics_05_by_model[
            model_a
        ][
            "roc_auc"
        ]
        -
        metrics_05_by_model[
            model_b
        ][
            "roc_auc"
        ]
    )

    point_ap_difference = (
        metrics_05_by_model[
            model_a
        ][
            "average_precision"
        ]
        -
        metrics_05_by_model[
            model_b
        ][
            "average_precision"
        ]
    )

    auc_differences = difference_bootstrap[
        (
            model_a,
            model_b,
        )
    ][
        "roc_auc"
    ]

    ap_differences = difference_bootstrap[
        (
            model_a,
            model_b,
        )
    ][
        "average_precision"
    ]

    auc_low, auc_high = percentile_interval(
        auc_differences
    )

    ap_low, ap_high = percentile_interval(
        ap_differences
    )

    difference_rows.append({
        "comparison": (
            f"{model_a} - {model_b}"
        ),
        "model_a": model_a,
        "model_b": model_b,
        "roc_auc_difference": float(
            point_auc_difference
        ),
        "roc_auc_difference_ci_low": auc_low,
        "roc_auc_difference_ci_high": auc_high,
        "roc_auc_bootstrap_p": (
            bootstrap_two_sided_pvalue(
                auc_differences
            )
        ),
        "average_precision_difference": float(
            point_ap_difference
        ),
        "average_precision_difference_ci_low": ap_low,
        "average_precision_difference_ci_high": ap_high,
        "average_precision_bootstrap_p": (
            bootstrap_two_sided_pvalue(
                ap_differences
            )
        ),
    })

difference_df = pd.DataFrame(
    difference_rows
)


# ============================================================
# Save combined predictions
# ============================================================

combined_output = combined[
    [
        "full_image_id",
        "patient_id",
        "binary_pathology",
        "label",
    ]
].copy()

combined_output[
    "cnn_probability"
] = model_probabilities[
    "CNN-only"
]

combined_output[
    "radiomics_probability"
] = model_probabilities[
    "Radiomics-only"
]

combined_output[
    "fusion_probability"
] = model_probabilities[
    "Fusion"
]

combined_output.to_csv(
    COMBINED_PREDICTIONS_FILE,
    index=False,
)


# ============================================================
# Save tables
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False,
)

model_interval_df.to_csv(
    BOOTSTRAP_MODEL_CSV,
    index=False,
)

difference_df.to_csv(
    BOOTSTRAP_DIFF_CSV,
    index=False,
)


# ============================================================
# Publication-ready LaTeX table
# ============================================================

latex_rows = []

for _, row in summary_df.iterrows():
    model_name = str(
        row[
            "model"
        ]
    )

    interval_row = model_interval_df[
        model_interval_df[
            "model"
        ] == model_name
    ].iloc[0]

    latex_rows.append(
        (
            f"{model_name} & "
            f"{row['roc_auc']:.4f} & "
            f"{interval_row['roc_auc_ci_low']:.4f}--"
            f"{interval_row['roc_auc_ci_high']:.4f} & "
            f"{row['average_precision']:.4f} & "
            f"{row['accuracy']:.4f} & "
            f"{row['recall_sensitivity']:.4f} & "
            f"{row['specificity']:.4f} & "
            f"{row['f1']:.4f} \\\\"
        )
    )

latex_text = r"""\begin{table}[htbp]
    \centering
    \caption{Comparison of CNN-only, radiomics-only, and multimodal
    fusion performance on the matched 414-mammogram test cohort at a
    classification threshold of 0.5. ROC AUC confidence intervals were
    estimated using paired stratified bootstrap resampling.}
    \label{tab:rq1_final_model_comparison}
    \small
    \begin{tabular}{lccccccc}
        \toprule
        Model & ROC AUC & 95\% CI & AP & Accuracy &
        Sensitivity & Specificity & F1 \\
        \midrule
""" + "\n".join(
    "        " + row
    for row in latex_rows
) + r"""
        \bottomrule
    \end{tabular}
\end{table}
"""

LATEX_TABLE_FILE.write_text(
    latex_text,
    encoding="utf-8",
)


# ============================================================
# Figures
# ============================================================

save_roc_figure(
    y_true,
    model_probabilities,
)

save_pr_figure(
    y_true,
    model_probabilities,
)

save_threshold_metric_figure(
    summary_df
)


# ============================================================
# Summary JSON
# ============================================================

summary_json = {
    "matched_test_mammograms": int(
        len(
            combined
        )
    ),
    "class_counts": {
        str(key): int(value)
        for key, value in combined[
            "binary_pathology"
        ].value_counts().items()
    },
    "bootstrap_replicates": int(
        N_BOOTSTRAP
    ),
    "bootstrap_method": (
        "paired stratified percentile bootstrap"
    ),
    "threshold_0_5_metrics": (
        summary_df.to_dict(
            orient="records"
        )
    ),
    "validation_selected_threshold_metrics": (
        selected_threshold_df.to_dict(
            orient="records"
        )
    ),
    "bootstrap_model_intervals": (
        model_interval_df.to_dict(
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
        summary_json,
        handle,
        indent=2,
    )


# ============================================================
# Console results
# ============================================================

print("\n")
print("=" * 72)
print("RQ1 COMPARISON COMPLETE")
print("=" * 72)

print(
    "\nBootstrap 95% confidence intervals:"
)

print(
    model_interval_df.to_string(
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

fusion_vs_cnn = difference_df[
    (
        difference_df[
            "model_a"
        ] == "Fusion"
    )
    &
    (
        difference_df[
            "model_b"
        ] == "CNN-only"
    )
].iloc[0]

print(
    "\nPrimary RQ1 comparison: Fusion - CNN-only"
)

print(
    "ROC AUC difference: "
    f"{fusion_vs_cnn['roc_auc_difference']:+.4f} "
    "["
    f"{fusion_vs_cnn['roc_auc_difference_ci_low']:+.4f}, "
    f"{fusion_vs_cnn['roc_auc_difference_ci_high']:+.4f}"
    "]"
)

print(
    "Bootstrap p-value: "
    f"{fusion_vs_cnn['roc_auc_bootstrap_p']:.4f}"
)

print(
    "Average precision difference: "
    f"{fusion_vs_cnn['average_precision_difference']:+.4f} "
    "["
    f"{fusion_vs_cnn['average_precision_difference_ci_low']:+.4f}, "
    f"{fusion_vs_cnn['average_precision_difference_ci_high']:+.4f}"
    "]"
)

print(
    "Bootstrap p-value: "
    f"{fusion_vs_cnn['average_precision_bootstrap_p']:.4f}"
)

print(
    "\nSaved comparison directory:"
)

print(
    OUTPUT_DIR.resolve()
)
