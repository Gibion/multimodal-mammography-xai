from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
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

INPUT_FILE = Path(
    "data/processed/cbis_ddsm/radiomics/feature_selection/"
    "cbis_ddsm_radiomics_selected_features.csv"
)

SELECTED_FEATURES_FILE = Path(
    "data/processed/cbis_ddsm/radiomics/feature_selection/"
    "selected_radiomics_features.csv"
)

OUTPUT_DIR = Path(
    "results/radiomics/radiomics_baseline"
)

MODEL_DIR = Path(
    "models/radiomics_baseline"
)

MODEL_FILE = MODEL_DIR / "logistic_regression.joblib"

METRICS_JSON = OUTPUT_DIR / "metrics_summary.json"

VALIDATION_PREDICTIONS_FILE = (
    OUTPUT_DIR / "validation_predictions.csv"
)

TEST_PREDICTIONS_FILE = (
    OUTPUT_DIR / "test_predictions.csv"
)

COEFFICIENTS_FILE = (
    OUTPUT_DIR / "radiomics_baseline_coefficients.csv"
)

DEFAULT_THRESHOLD = 0.5
RANDOM_SEED = 42

# The feature-selection stage already chose a regularisation strength
# using training-only CV. The baseline classifier uses that selected C.
# Update automatically from the feature-selection summary if present.
FEATURE_SELECTION_SUMMARY = Path(
    "data/processed/cbis_ddsm/radiomics/feature_selection/"
    "radiomics_feature_selection_summary.json"
)

FALLBACK_C = 0.428133

MAX_ITER = 5000
SOLVER = "liblinear"


# ============================================================
# Helpers
# ============================================================

def label_to_int(value):
    text = str(value).strip().upper()

    if text == "MALIGNANT":
        return 1

    if text == "BENIGN":
        return 0

    raise ValueError(
        f"Unexpected binary_pathology value: {value}"
    )


def load_selected_feature_columns(frame):
    selected_columns = [
        column
        for column in frame.columns
        if column.startswith(
            "selected_scaled_"
        )
    ]

    if not selected_columns:
        raise ValueError(
            "No selected_scaled_* feature columns were found."
        )

    return selected_columns


def choose_validation_threshold(
    y_true,
    probabilities
):
    """
    Select threshold using validation F1 only.

    The test set is never used for threshold selection.
    """

    thresholds = np.unique(
        np.concatenate(
            [
                np.array([0.0, 0.5, 1.0]),
                probabilities,
            ]
        )
    )

    best_threshold = DEFAULT_THRESHOLD
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)

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
        probabilities >= threshold
    ).astype(int)

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
        if (tn + fp) > 0
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


def save_prediction_file(
    frame,
    probabilities,
    threshold,
    output_path,
):
    output_columns = [
        column
        for column in [
            "full_image_id",
            "patient_id",
            "split",
            "binary_pathology",
            "n_lesions",
            "radiomics_lesion_group",
            "preprocessed_path",
            "full_mammogram_path",
        ]
        if column in frame.columns
    ]

    output = frame[
        output_columns
    ].copy()

    output[
        "label"
    ] = frame[
        "binary_pathology"
    ].apply(
        label_to_int
    ).to_numpy(
        dtype=int
    )

    output[
        "predicted_probability"
    ] = probabilities

    output[
        "predicted_label_default"
    ] = (
        probabilities
        >= DEFAULT_THRESHOLD
    ).astype(int)

    output[
        "predicted_label_selected"
    ] = (
        probabilities
        >= threshold
    ).astype(int)

    output.to_csv(
        output_path,
        index=False,
    )


def save_roc_curve(
    y_validation,
    validation_probabilities,
    y_test,
    test_probabilities,
):
    val_fpr, val_tpr, _ = roc_curve(
        y_validation,
        validation_probabilities,
    )

    test_fpr, test_tpr, _ = roc_curve(
        y_test,
        test_probabilities,
    )

    val_auc = roc_auc_score(
        y_validation,
        validation_probabilities,
    )

    test_auc = roc_auc_score(
        y_test,
        test_probabilities,
    )

    fig = plt.figure(
        figsize=(
            7,
            6,
        )
    )

    plt.plot(
        val_fpr,
        val_tpr,
        label=f"Validation AUC = {val_auc:.3f}",
    )

    plt.plot(
        test_fpr,
        test_tpr,
        label=f"Test AUC = {test_auc:.3f}",
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
        "Radiomics-Only Logistic Regression ROC Curves"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_precision_recall_curve(
    y_validation,
    validation_probabilities,
    y_test,
    test_probabilities,
):
    val_precision, val_recall, _ = (
        precision_recall_curve(
            y_validation,
            validation_probabilities,
        )
    )

    test_precision, test_recall, _ = (
        precision_recall_curve(
            y_test,
            test_probabilities,
        )
    )

    val_ap = average_precision_score(
        y_validation,
        validation_probabilities,
    )

    test_ap = average_precision_score(
        y_test,
        test_probabilities,
    )

    fig = plt.figure(
        figsize=(
            7,
            6,
        )
    )

    plt.plot(
        val_recall,
        val_precision,
        label=f"Validation AP = {val_ap:.3f}",
    )

    plt.plot(
        test_recall,
        test_precision,
        label=f"Test AP = {test_ap:.3f}",
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Radiomics-Only Logistic Regression Precision-Recall Curves"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "precision_recall_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_confusion_matrix_figure(
    metrics_default,
    metrics_selected,
):
    matrices = [
        np.array(
            [
                [
                    metrics_default[
                        "tn"
                    ],
                    metrics_default[
                        "fp"
                    ],
                ],
                [
                    metrics_default[
                        "fn"
                    ],
                    metrics_default[
                        "tp"
                    ],
                ],
            ]
        ),
        np.array(
            [
                [
                    metrics_selected[
                        "tn"
                    ],
                    metrics_selected[
                        "fp"
                    ],
                ],
                [
                    metrics_selected[
                        "fn"
                    ],
                    metrics_selected[
                        "tp"
                    ],
                ],
            ]
        ),
    ]

    titles = [
        (
            "Test Confusion Matrix "
            f"(threshold={DEFAULT_THRESHOLD:.3f})"
        ),
        (
            "Test Confusion Matrix "
            f"(threshold={metrics_selected['threshold']:.3f})"
        ),
    ]

    for matrix, title, filename in zip(
        matrices,
        titles,
        [
            "confusion_matrix_test_threshold_0_5.png",
            "confusion_matrix_test_selected_threshold.png",
        ],
    ):
        fig = plt.figure(
            figsize=(
                5.5,
                5,
            )
        )

        plt.imshow(
            matrix,
            aspect="equal",
        )

        for row in range(
            2
        ):
            for column in range(
                2
            ):
                plt.text(
                    column,
                    row,
                    str(
                        matrix[
                            row,
                            column
                        ]
                    ),
                    ha="center",
                    va="center",
                    fontsize=14,
                )

        plt.xticks(
            [
                0,
                1,
            ],
            [
                "Predicted Benign",
                "Predicted Malignant",
            ],
            rotation=20,
        )

        plt.yticks(
            [
                0,
                1,
            ],
            [
                "Actual Benign",
                "Actual Malignant",
            ],
        )

        plt.title(
            title
        )

        plt.tight_layout()

        fig.savefig(
            OUTPUT_DIR
            / filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )


# ============================================================
# Validate inputs
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Selected-feature dataset not found:\n"
        f"{INPUT_FILE.resolve()}"
    )

df = pd.read_csv(
    INPUT_FILE
)

selected_columns = (
    load_selected_feature_columns(
        df
    )
)

if len(
    selected_columns
) != 28:
    print(
        "\nWARNING: Expected 28 selected radiomics features, "
        f"but found {len(selected_columns)}."
    )

for column in selected_columns:
    df[
        column
    ] = pd.to_numeric(
        df[
            column
        ],
        errors="coerce"
    )

if df[
    selected_columns
].isna().any().any():
    raise ValueError(
        "Missing values were found in selected predictors."
    )

if np.isinf(
    df[
        selected_columns
    ].to_numpy(
        dtype=float
    )
).any():
    raise ValueError(
        "Infinite values were found in selected predictors."
    )


# ============================================================
# Split data
# ============================================================

train_df = df[
    df[
        "split"
    ] == "train"
].copy()

validation_df = df[
    df[
        "split"
    ] == "validation"
].copy()

test_df = df[
    df[
        "split"
    ] == "test"
].copy()

X_train = train_df[
    selected_columns
].to_numpy(
    dtype=float
)

X_validation = validation_df[
    selected_columns
].to_numpy(
    dtype=float
)

X_test = test_df[
    selected_columns
].to_numpy(
    dtype=float
)

y_train = train_df[
    "binary_pathology"
].apply(
    label_to_int
).to_numpy(
    dtype=int
)

y_validation = validation_df[
    "binary_pathology"
].apply(
    label_to_int
).to_numpy(
    dtype=int
)

y_test = test_df[
    "binary_pathology"
].apply(
    label_to_int
).to_numpy(
    dtype=int
)


# ============================================================
# Read C chosen during training-only feature selection
# ============================================================

regularisation_c = FALLBACK_C

if FEATURE_SELECTION_SUMMARY.exists():
    with FEATURE_SELECTION_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as handle:
        feature_selection_summary = json.load(
            handle
        )

    regularisation_c = float(
        feature_selection_summary[
            "supervised_selector"
        ][
            "best_C"
        ]
    )


# ============================================================
# Train radiomics-only baseline
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 72)
print("RADIOMICS-ONLY BASELINE: LOGISTIC REGRESSION")
print("=" * 72)

print(
    f"\nTraining mammograms: "
    f"{len(train_df):,}"
)

print(
    f"Validation mammograms: "
    f"{len(validation_df):,}"
)

print(
    f"Test mammograms: "
    f"{len(test_df):,}"
)

print(
    f"\nSelected radiomics features: "
    f"{len(selected_columns)}"
)

print(
    f"L1 inverse regularisation C: "
    f"{regularisation_c:.6g}"
)

print(
    "\nTraining class counts:"
)

print(
    train_df[
        "binary_pathology"
    ].value_counts()
)

model = LogisticRegression(
    penalty="l1",
    solver=SOLVER,
    C=regularisation_c,
    max_iter=MAX_ITER,
    random_state=RANDOM_SEED,
)

model.fit(
    X_train,
    y_train
)

joblib.dump(
    model,
    MODEL_FILE
)


# ============================================================
# Predict validation and test
# ============================================================

validation_probabilities = model.predict_proba(
    X_validation
)[
    :,
    1
]

test_probabilities = model.predict_proba(
    X_test
)[
    :,
    1
]


# ============================================================
# Select threshold on validation only
# ============================================================

(
    selected_threshold,
    selected_validation_f1,
) = choose_validation_threshold(
    y_validation,
    validation_probabilities,
)

print(
    "\nThreshold selection:"
)

print(
    f"Default threshold: "
    f"{DEFAULT_THRESHOLD:.3f}"
)

print(
    f"Validation-selected threshold "
    f"(max F1): "
    f"{selected_threshold:.3f}"
)

print(
    f"Validation F1 at selected threshold: "
    f"{selected_validation_f1:.4f}"
)


# ============================================================
# Metrics
# ============================================================

validation_default_metrics = (
    calculate_metrics(
        y_validation,
        validation_probabilities,
        DEFAULT_THRESHOLD,
    )
)

validation_selected_metrics = (
    calculate_metrics(
        y_validation,
        validation_probabilities,
        selected_threshold,
    )
)

test_default_metrics = (
    calculate_metrics(
        y_test,
        test_probabilities,
        DEFAULT_THRESHOLD,
    )
)

test_selected_metrics = (
    calculate_metrics(
        y_test,
        test_probabilities,
        selected_threshold,
    )
)

print_metrics(
    "Validation metrics at threshold 0.5",
    validation_default_metrics,
)

print_metrics(
    "Validation metrics at selected threshold",
    validation_selected_metrics,
)

print_metrics(
    "Test metrics at threshold 0.5",
    test_default_metrics,
)

print_metrics(
    "Test metrics at validation-selected threshold",
    test_selected_metrics,
)


# ============================================================
# Save predictions
# ============================================================

save_prediction_file(
    validation_df,
    validation_probabilities,
    selected_threshold,
    VALIDATION_PREDICTIONS_FILE,
)

save_prediction_file(
    test_df,
    test_probabilities,
    selected_threshold,
    TEST_PREDICTIONS_FILE,
)


# ============================================================
# Coefficients
# ============================================================

coefficient_df = pd.DataFrame({
    "selected_scaled_feature": selected_columns,
    "coefficient": np.ravel(
        model.coef_
    ),
})

coefficient_df[
    "absolute_coefficient"
] = np.abs(
    coefficient_df[
        "coefficient"
    ]
)

coefficient_df = coefficient_df.sort_values(
    "absolute_coefficient",
    ascending=False,
).reset_index(
    drop=True
)

coefficient_df[
    "rank"
] = np.arange(
    1,
    len(
        coefficient_df
    )
    + 1
)

coefficient_df.to_csv(
    COEFFICIENTS_FILE,
    index=False,
)


# ============================================================
# Figures
# ============================================================

save_roc_curve(
    y_validation,
    validation_probabilities,
    y_test,
    test_probabilities,
)

save_precision_recall_curve(
    y_validation,
    validation_probabilities,
    y_test,
    test_probabilities,
)

save_confusion_matrix_figure(
    test_default_metrics,
    test_selected_metrics,
)


# ============================================================
# Save summary
# ============================================================

summary = {
    "model": "L1 logistic regression",
    "feature_count": int(
        len(
            selected_columns
        )
    ),
    "regularisation_C": float(
        regularisation_c
    ),
    "split_counts": {
        "train": int(
            len(
                train_df
            )
        ),
        "validation": int(
            len(
                validation_df
            )
        ),
        "test": int(
            len(
                test_df
            )
        ),
    },
    "threshold_selection": {
        "method": "validation maximum F1",
        "selected_threshold": float(
            selected_threshold
        ),
        "validation_f1": float(
            selected_validation_f1
        ),
    },
    "validation_threshold_0_5": (
        validation_default_metrics
    ),
    "validation_selected_threshold": (
        validation_selected_metrics
    ),
    "test_threshold_0_5": (
        test_default_metrics
    ),
    "test_selected_threshold": (
        test_selected_metrics
    ),
}

with METRICS_JSON.open(
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
print("RADIOMICS-ONLY BASELINE COMPLETE")
print("=" * 72)

print(
    f"\nValidation-selected threshold: "
    f"{selected_threshold:.3f}"
)

print(
    f"Test ROC AUC: "
    f"{test_default_metrics['roc_auc']:.4f}"
)

print(
    f"Test average precision: "
    f"{test_default_metrics['average_precision']:.4f}"
)

print(
    "\nTest threshold 0.5:"
)

print(
    f"accuracy    = "
    f"{test_default_metrics['accuracy']:.4f}"
)

print(
    f"precision   = "
    f"{test_default_metrics['precision']:.4f}"
)

print(
    f"recall      = "
    f"{test_default_metrics['recall_sensitivity']:.4f}"
)

print(
    f"specificity = "
    f"{test_default_metrics['specificity']:.4f}"
)

print(
    f"F1          = "
    f"{test_default_metrics['f1']:.4f}"
)

print(
    "\nTest validation-selected threshold:"
)

print(
    f"threshold   = "
    f"{selected_threshold:.3f}"
)

print(
    f"accuracy    = "
    f"{test_selected_metrics['accuracy']:.4f}"
)

print(
    f"precision   = "
    f"{test_selected_metrics['precision']:.4f}"
)

print(
    f"recall      = "
    f"{test_selected_metrics['recall_sensitivity']:.4f}"
)

print(
    f"specificity = "
    f"{test_selected_metrics['specificity']:.4f}"
)

print(
    f"F1          = "
    f"{test_selected_metrics['f1']:.4f}"
)

print(
    "\nSaved model:"
)

print(
    MODEL_FILE.resolve()
)

print(
    "\nSaved results:"
)

print(
    OUTPUT_DIR.resolve()
)
