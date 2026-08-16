from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.exceptions import ConvergenceWarning
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
from sklearn.model_selection import StratifiedKFold, GridSearchCV


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path(
    "data/processed/cbis_ddsm/fusion/"
    "cbis_ddsm_multimodal_fusion_features.csv"
)

OUTPUT_DIR = Path(
    "results/multimodal_fusion/logistic_regression"
)

MODEL_DIR = Path(
    "models/multimodal_fusion_logistic_regression"
)

RANDOM_STATE = 42
CV_FOLDS = 5

# Broad logarithmic search. This is fit using TRAINING data only.
C_GRID = np.logspace(-4, 2, 15)

# Use L2 regularisation for the primary concatenated-feature fusion
# baseline. Feature selection has already occurred upstream for
# radiomics, and constant CNN dimensions were removed upstream.
PENALTY = "l2"
SOLVER = "liblinear"
MAX_ITER = 5000


# ============================================================
# Helpers
# ============================================================

def encode_labels(series):
    mapping = {
        "BENIGN": 0,
        "MALIGNANT": 1,
        0: 0,
        1: 1,
        "0": 0,
        "1": 1,
    }

    encoded = series.map(mapping)

    if encoded.isna().any():
        bad = sorted(
            series[
                encoded.isna()
            ].astype(str).unique()
        )
        raise ValueError(
            "Unexpected binary_pathology values: "
            + ", ".join(bad)
        )

    return encoded.astype(int).to_numpy()


def select_f1_threshold(y_true, probabilities):
    """
    Select the threshold that maximises F1 on validation data only.
    Ties are resolved by choosing the threshold closest to 0.5.
    """
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    # precision/recall contain one more value than thresholds.
    precision = precision[:-1]
    recall = recall[:-1]

    denominator = precision + recall

    f1_values = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(
            denominator,
            dtype=float,
        ),
        where=denominator > 0,
    )

    best_f1 = float(
        np.max(
            f1_values
        )
    )

    candidate_indices = np.flatnonzero(
        np.isclose(
            f1_values,
            best_f1,
            rtol=0.0,
            atol=1e-12,
        )
    )

    if len(candidate_indices) > 1:
        best_index = candidate_indices[
            np.argmin(
                np.abs(
                    thresholds[
                        candidate_indices
                    ] - 0.5
                )
            )
        ]
    else:
        best_index = int(
            candidate_indices[0]
        )

    return (
        float(
            thresholds[
                best_index
            ]
        ),
        best_f1,
    )


def calculate_metrics(
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

    metrics = {
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

    return metrics


def print_metrics(title, metrics):
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
            f"{key:<20}: "
            f"{metrics[key]:.4f}"
        )

    print(
        "confusion matrix: "
        f"TN={metrics['tn']}, "
        f"FP={metrics['fp']}, "
        f"FN={metrics['fn']}, "
        f"TP={metrics['tp']}"
    )


def save_predictions(
    frame,
    probabilities,
    threshold,
    output_path,
):
    output = frame[
        [
            column
            for column in [
                "full_image_id",
                "patient_id",
                "split",
                "binary_pathology",
            ]
            if column in frame.columns
        ]
    ].copy()

    y_true = encode_labels(
        frame[
            "binary_pathology"
        ]
    )

    output[
        "true_label"
    ] = y_true

    output[
        "predicted_probability"
    ] = probabilities

    output[
        "prediction_threshold_0_5"
    ] = (
        probabilities >= 0.5
    ).astype(int)

    output[
        "prediction_validation_selected"
    ] = (
        probabilities >= threshold
    ).astype(int)

    output.to_csv(
        output_path,
        index=False,
    )


def plot_roc(
    y_validation,
    p_validation,
    y_test,
    p_test,
    output_path,
):
    fpr_val, tpr_val, _ = roc_curve(
        y_validation,
        p_validation,
    )
    fpr_test, tpr_test, _ = roc_curve(
        y_test,
        p_test,
    )

    auc_val = roc_auc_score(
        y_validation,
        p_validation,
    )
    auc_test = roc_auc_score(
        y_test,
        p_test,
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        fpr_val,
        tpr_val,
        label=(
            "Validation "
            f"(AUC = {auc_val:.3f})"
        ),
    )

    plt.plot(
        fpr_test,
        tpr_test,
        label=(
            "Test "
            f"(AUC = {auc_test:.3f})"
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
        "Multimodal Fusion ROC Curves"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_precision_recall(
    y_validation,
    p_validation,
    y_test,
    p_test,
    output_path,
):
    precision_val, recall_val, _ = (
        precision_recall_curve(
            y_validation,
            p_validation,
        )
    )

    precision_test, recall_test, _ = (
        precision_recall_curve(
            y_test,
            p_test,
        )
    )

    ap_val = average_precision_score(
        y_validation,
        p_validation,
    )
    ap_test = average_precision_score(
        y_test,
        p_test,
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        recall_val,
        precision_val,
        label=(
            "Validation "
            f"(AP = {ap_val:.3f})"
        ),
    )

    plt.plot(
        recall_test,
        precision_test,
        label=(
            "Test "
            f"(AP = {ap_test:.3f})"
        ),
    )

    plt.xlabel(
        "Recall"
    )
    plt.ylabel(
        "Precision"
    )
    plt.title(
        "Multimodal Fusion Precision-Recall Curves"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ============================================================
# Load prepared fusion data
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Prepared fusion dataset not found:\n"
        f"{INPUT_FILE.resolve()}"
    )

df = pd.read_csv(
    INPUT_FILE
)

required_columns = {
    "full_image_id",
    "patient_id",
    "split",
    "binary_pathology",
}

missing_required = (
    required_columns
    - set(
        df.columns
    )
)

if missing_required:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(
            sorted(
                missing_required
            )
        )
    )

if df[
    "full_image_id"
].duplicated().any():
    raise ValueError(
        "Duplicate full_image_id rows found."
    )

cnn_columns = [
    column
    for column in df.columns
    if column.startswith(
        "cnn_scaled_"
    )
]

radiomics_columns = [
    column
    for column in df.columns
    if column.startswith(
        "radiomics_selected_"
    )
]

feature_columns = (
    cnn_columns
    + radiomics_columns
)

if not cnn_columns:
    raise ValueError(
        "No cnn_scaled_* columns found."
    )

if not radiomics_columns:
    raise ValueError(
        "No radiomics_selected_* columns found."
    )

X_all = df[
    feature_columns
].to_numpy(
    dtype=float
)

if np.isnan(
    X_all
).any():
    raise ValueError(
        "NaN values found in fusion predictors."
    )

if np.isinf(
    X_all
).any():
    raise ValueError(
        "Infinite values found in fusion predictors."
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

if (
    len(train_df) == 0
    or len(validation_df) == 0
    or len(test_df) == 0
):
    raise ValueError(
        "Train, validation, and test splits are required."
    )

X_train = train_df[
    feature_columns
].to_numpy(
    dtype=float
)

X_validation = validation_df[
    feature_columns
].to_numpy(
    dtype=float
)

X_test = test_df[
    feature_columns
].to_numpy(
    dtype=float
)

y_train = encode_labels(
    train_df[
        "binary_pathology"
    ]
)

y_validation = encode_labels(
    validation_df[
        "binary_pathology"
    ]
)

y_test = encode_labels(
    test_df[
        "binary_pathology"
    ]
)


# ============================================================
# Console overview
# ============================================================

print("=" * 72)
print("MULTIMODAL FUSION BASELINE: LOGISTIC REGRESSION")
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
    f"\nCNN predictors: "
    f"{len(cnn_columns)}"
)

print(
    f"Radiomics predictors: "
    f"{len(radiomics_columns)}"
)

print(
    f"Total fusion predictors: "
    f"{len(feature_columns)}"
)

print(
    "\nTraining class counts:"
)

print(
    train_df[
        "binary_pathology"
    ].value_counts()
)


# ============================================================
# Training-only cross-validation for C
# ============================================================

base_model = LogisticRegression(
    penalty=PENALTY,
    solver=SOLVER,
    max_iter=MAX_ITER,
    class_weight=None,
    random_state=RANDOM_STATE,
)

cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

parameter_grid = {
    "C": C_GRID,
}

grid = GridSearchCV(
    estimator=base_model,
    param_grid=parameter_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=True,
)

print(
    "\nSelecting regularisation strength "
    "using training-only stratified "
    f"{CV_FOLDS}-fold CV..."
)

with warnings.catch_warnings():
    warnings.simplefilter(
        "ignore",
        category=ConvergenceWarning,
    )

    grid.fit(
        X_train,
        y_train,
    )

best_model = grid.best_estimator_
best_c = float(
    grid.best_params_[
        "C"
    ]
)

print(
    f"Best L2 C: "
    f"{best_c:.6f}"
)

print(
    f"Best mean training-CV ROC AUC: "
    f"{grid.best_score_:.4f}"
)


# ============================================================
# Save CV results
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

cv_results = pd.DataFrame(
    grid.cv_results_
)

cv_results.to_csv(
    OUTPUT_DIR
    / "fusion_logistic_regression_cv_results.csv",
    index=False,
)


# ============================================================
# Predict validation and test sets
# ============================================================

validation_probabilities = (
    best_model.predict_proba(
        X_validation
    )[:, 1]
)

test_probabilities = (
    best_model.predict_proba(
        X_test
    )[:, 1]
)


# ============================================================
# Validation-only threshold selection
# ============================================================

selected_threshold, validation_best_f1 = (
    select_f1_threshold(
        y_validation,
        validation_probabilities,
    )
)

print(
    "\nThreshold selection:"
)

print(
    "Default threshold: 0.500"
)

print(
    "Validation-selected threshold "
    f"(max F1): {selected_threshold:.3f}"
)

print(
    "Validation F1 at selected threshold: "
    f"{validation_best_f1:.4f}"
)


# ============================================================
# Metrics
# ============================================================

validation_metrics_05 = calculate_metrics(
    y_validation,
    validation_probabilities,
    0.5,
)

validation_metrics_selected = calculate_metrics(
    y_validation,
    validation_probabilities,
    selected_threshold,
)

test_metrics_05 = calculate_metrics(
    y_test,
    test_probabilities,
    0.5,
)

test_metrics_selected = calculate_metrics(
    y_test,
    test_probabilities,
    selected_threshold,
)

print_metrics(
    "Validation metrics at threshold 0.5",
    validation_metrics_05,
)

print_metrics(
    "Validation metrics at selected threshold",
    validation_metrics_selected,
)

print_metrics(
    "Test metrics at threshold 0.5",
    test_metrics_05,
)

print_metrics(
    "Test metrics at validation-selected threshold",
    test_metrics_selected,
)


# ============================================================
# Save predictions
# ============================================================

save_predictions(
    validation_df,
    validation_probabilities,
    selected_threshold,
    OUTPUT_DIR
    / "validation_predictions.csv",
)

save_predictions(
    test_df,
    test_probabilities,
    selected_threshold,
    OUTPUT_DIR
    / "test_predictions.csv",
)


# ============================================================
# Save model
# ============================================================

joblib.dump(
    best_model,
    MODEL_DIR
    / "best_model.joblib",
)

with (
    MODEL_DIR
    / "feature_columns.json"
).open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        {
            "cnn_features": (
                cnn_columns
            ),
            "radiomics_features": (
                radiomics_columns
            ),
            "all_features": (
                feature_columns
            ),
        },
        handle,
        indent=2,
    )


# ============================================================
# Save figures
# ============================================================

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

plot_roc(
    y_validation,
    validation_probabilities,
    y_test,
    test_probabilities,
    FIGURE_DIR
    / "fusion_roc_curves.png",
)

plot_precision_recall(
    y_validation,
    validation_probabilities,
    y_test,
    test_probabilities,
    FIGURE_DIR
    / "fusion_precision_recall_curves.png",
)


# ============================================================
# Save summary JSON
# ============================================================

summary = {
    "model": (
        "LogisticRegression"
    ),
    "fusion_strategy": (
        "early feature concatenation"
    ),
    "penalty": PENALTY,
    "solver": SOLVER,
    "cv_folds": CV_FOLDS,
    "cv_scoring": "roc_auc",
    "best_C": best_c,
    "best_training_cv_roc_auc": float(
        grid.best_score_
    ),
    "training_mammograms": int(
        len(train_df)
    ),
    "validation_mammograms": int(
        len(validation_df)
    ),
    "test_mammograms": int(
        len(test_df)
    ),
    "cnn_feature_dimension": int(
        len(cnn_columns)
    ),
    "radiomics_feature_dimension": int(
        len(radiomics_columns)
    ),
    "fusion_feature_dimension": int(
        len(feature_columns)
    ),
    "validation_selected_threshold": float(
        selected_threshold
    ),
    "validation_f1_at_selected_threshold": float(
        validation_best_f1
    ),
    "validation_threshold_0_5": (
        validation_metrics_05
    ),
    "validation_selected_threshold_metrics": (
        validation_metrics_selected
    ),
    "test_threshold_0_5": (
        test_metrics_05
    ),
    "test_validation_selected_threshold": (
        test_metrics_selected
    ),
}

with (
    OUTPUT_DIR
    / "fusion_logistic_regression_summary.json"
).open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        summary,
        handle,
        indent=2,
    )


# ============================================================
# Final concise summary
# ============================================================

print("\n")
print("=" * 72)
print("MULTIMODAL FUSION BASELINE COMPLETE")
print("=" * 72)

print(
    f"\nBest L2 C: "
    f"{best_c:.6f}"
)

print(
    "Validation-selected threshold: "
    f"{selected_threshold:.3f}"
)

print(
    "Test ROC AUC: "
    f"{test_metrics_05['roc_auc']:.4f}"
)

print(
    "Test average precision: "
    f"{test_metrics_05['average_precision']:.4f}"
)

print(
    "\nTest threshold 0.5:"
)

print(
    "accuracy    = "
    f"{test_metrics_05['accuracy']:.4f}"
)

print(
    "precision   = "
    f"{test_metrics_05['precision']:.4f}"
)

print(
    "recall      = "
    f"{test_metrics_05['recall_sensitivity']:.4f}"
)

print(
    "specificity = "
    f"{test_metrics_05['specificity']:.4f}"
)

print(
    "F1          = "
    f"{test_metrics_05['f1']:.4f}"
)

print(
    "\nTest validation-selected threshold:"
)

print(
    "threshold   = "
    f"{selected_threshold:.3f}"
)

print(
    "accuracy    = "
    f"{test_metrics_selected['accuracy']:.4f}"
)

print(
    "precision   = "
    f"{test_metrics_selected['precision']:.4f}"
)

print(
    "recall      = "
    f"{test_metrics_selected['recall_sensitivity']:.4f}"
)

print(
    "specificity = "
    f"{test_metrics_selected['specificity']:.4f}"
)

print(
    "F1          = "
    f"{test_metrics_selected['f1']:.4f}"
)

print(
    "\nSaved results:"
)

print(
    OUTPUT_DIR.resolve()
)

print(
    "\nSaved model:"
)

print(
    (
        MODEL_DIR
        / "best_model.joblib"
    ).resolve()
)
