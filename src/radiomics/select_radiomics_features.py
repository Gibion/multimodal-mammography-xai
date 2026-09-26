from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path.cwd()

INPUT_FILE = Path(
    "data/processed/cbis_ddsm/radiomics/"
    "cbis_ddsm_radiomics_mammogram_features.csv"
)

OUTPUT_DIR = Path(
    "data/processed/cbis_ddsm/radiomics/feature_selection"
)

FIGURE_DIR = Path(
    "results/radiomics/feature_selection"
)

SELECTED_DATA_FILE = OUTPUT_DIR / "cbis_ddsm_radiomics_selected_features.csv"
SELECTED_FEATURES_FILE = OUTPUT_DIR / "selected_radiomics_features.csv"
CORRELATION_DROPS_FILE = OUTPUT_DIR / "correlation_filtered_features.csv"
COEFFICIENTS_FILE = OUTPUT_DIR / "l1_feature_selection_coefficients.csv"
SUMMARY_FILE = OUTPUT_DIR / "radiomics_feature_selection_summary.json"
SCALER_FILE = OUTPUT_DIR / "radiomics_standard_scaler.joblib"
L1_SELECTOR_FILE = OUTPUT_DIR / "radiomics_l1_selector.joblib"

FEATURE_PREFIX = "original_"
INCLUDE_N_LESIONS = True
CORRELATION_METHOD = "spearman"
CORRELATION_THRESHOLD = 0.90
CV_FOLDS = 5
RANDOM_SEED = 42
C_VALUES = np.logspace(-3, 2, 20)
MAX_ITER = 5000
TOL = 1e-4
COEFFICIENT_EPSILON = 1e-8


def binary_label(value):
    text = str(value).strip().upper()
    if text == "MALIGNANT":
        return 1
    if text == "BENIGN":
        return 0
    raise ValueError(f"Unexpected binary_pathology value: {value}")


def get_predictor_columns(frame):
    predictors = [
        c for c in frame.columns
        if (
            c.startswith(FEATURE_PREFIX)
            and (c.endswith("_mean") or c.endswith("_max"))
        )
    ]
    if INCLUDE_N_LESIONS:
        predictors.append("n_lesions")
    return predictors


def validate_split_integrity(frame):
    required = ["patient_id", "split", "binary_pathology", "full_image_id"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(
            "Input file is missing required columns: "
            + ", ".join(missing)
        )

    if frame["full_image_id"].duplicated().any():
        raise ValueError("Duplicate full_image_id rows were found.")

    train_patients = set(
        frame.loc[frame["split"] == "train", "patient_id"].astype(str)
    )
    validation_patients = set(
        frame.loc[frame["split"] == "validation", "patient_id"].astype(str)
    )
    test_patients = set(
        frame.loc[frame["split"] == "test", "patient_id"].astype(str)
    )

    overlaps = {
        "train_validation": len(train_patients & validation_patients),
        "train_test": len(train_patients & test_patients),
        "validation_test": len(validation_patients & test_patients),
    }

    if any(v > 0 for v in overlaps.values()):
        raise ValueError(
            "Participant overlap detected between splits: "
            + str(overlaps)
        )

    return overlaps


def remove_constant_features(train_frame, columns):
    kept, dropped = [], []
    for column in columns:
        if train_frame[column].nunique(dropna=True) <= 1:
            dropped.append(column)
        else:
            kept.append(column)
    return kept, dropped


def correlation_filter(train_frame, columns, threshold):
    corr = train_frame[columns].corr(
        method=CORRELATION_METHOD
    ).abs()

    kept = []
    dropped = []

    for feature in columns:
        conflict = None
        conflict_value = None

        for retained in kept:
            value = corr.loc[feature, retained]
            if pd.notna(value) and value > threshold:
                conflict = retained
                conflict_value = float(value)
                break

        if conflict is None:
            kept.append(feature)
        else:
            dropped.append({
                "dropped_feature": feature,
                "retained_correlated_feature": conflict,
                "absolute_spearman_rho": conflict_value,
            })

    return kept, pd.DataFrame(dropped), corr


def save_correlation_heatmap(matrix, output_path, title):
    if matrix.empty:
        return

    size = max(8, min(24, 0.22 * len(matrix)))
    fig = plt.figure(figsize=(size, size))

    plt.imshow(
        matrix.to_numpy(),
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    if len(matrix) <= 40:
        positions = np.arange(len(matrix))
        labels = [
            c.replace("original_", "")
            .replace("_mean", " mean")
            .replace("_max", " max")
            for c in matrix.columns
        ]
        plt.xticks(positions, labels, rotation=90, fontsize=6)
        plt.yticks(positions, labels, fontsize=6)
    else:
        plt.xticks([])
        plt.yticks([])

    plt.colorbar(label="Absolute Spearman correlation")
    plt.title(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE.resolve()}"
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 72)
print("RADIOMICS FEATURE PREPROCESSING AND SELECTION")
print("=" * 72)

print(f"\nMammogram rows: {len(df):,}")

overlaps = validate_split_integrity(df)

predictor_columns = get_predictor_columns(df)

print(f"Initial predictor candidates: {len(predictor_columns):,}")

for column in predictor_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

problem_rows = []
for column in predictor_columns:
    values = df[column].to_numpy(dtype=float)
    n_nan = int(np.isnan(values).sum())
    n_inf = int(np.isinf(values).sum())
    if n_nan or n_inf:
        problem_rows.append({
            "feature": column,
            "n_nan": n_nan,
            "n_inf": n_inf,
        })

if problem_rows:
    raise ValueError(
        "Non-finite predictor values found:\n"
        + pd.DataFrame(problem_rows).to_string(index=False)
    )

train_df = df[df["split"] == "train"].copy()
validation_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

y_train = train_df["binary_pathology"].apply(binary_label).to_numpy(dtype=int)
y_validation = validation_df["binary_pathology"].apply(binary_label).to_numpy(dtype=int)
y_test = test_df["binary_pathology"].apply(binary_label).to_numpy(dtype=int)

print("\nSplit counts:")
print(df["split"].value_counts())

print("\nTraining class counts:")
print(train_df["binary_pathology"].value_counts())

nonconstant_features, constant_features = remove_constant_features(
    train_df,
    predictor_columns
)

print(
    f"\nTraining-constant features removed: "
    f"{len(constant_features)}"
)
print(
    f"Features after constant check: "
    f"{len(nonconstant_features)}"
)

(
    correlation_retained_features,
    correlation_drops_df,
    full_correlation_matrix,
) = correlation_filter(
    train_df,
    nonconstant_features,
    CORRELATION_THRESHOLD
)

print(
    f"\nCorrelation threshold: "
    f"|rho| > {CORRELATION_THRESHOLD:.2f}"
)
print(
    f"Features removed by correlation filter: "
    f"{len(correlation_drops_df)}"
)
print(
    f"Features retained after correlation filter: "
    f"{len(correlation_retained_features)}"
)

X_train_raw = train_df[
    correlation_retained_features
].to_numpy(dtype=float)

X_validation_raw = validation_df[
    correlation_retained_features
].to_numpy(dtype=float)

X_test_raw = test_df[
    correlation_retained_features
].to_numpy(dtype=float)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_validation = scaler.transform(X_validation_raw)
X_test = scaler.transform(X_test_raw)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(scaler, SCALER_FILE)

cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_SEED,
)

selector = LogisticRegressionCV(
    Cs=C_VALUES,
    cv=cv,
    penalty="l1",
    solver="liblinear",
    scoring="roc_auc",
    max_iter=MAX_ITER,
    tol=TOL,
    random_state=RANDOM_SEED,
    refit=True,
    n_jobs=-1,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    selector.fit(X_train, y_train)

joblib.dump(selector, L1_SELECTOR_FILE)

best_c = float(np.ravel(selector.C_)[0])
coefficients = np.ravel(selector.coef_)

coefficient_df = pd.DataFrame({
    "feature": correlation_retained_features,
    "coefficient": coefficients,
    "absolute_coefficient": np.abs(coefficients),
})

coefficient_df["selected"] = (
    coefficient_df["absolute_coefficient"] > COEFFICIENT_EPSILON
)

coefficient_df = coefficient_df.sort_values(
    "absolute_coefficient",
    ascending=False
).reset_index(drop=True)

selected_features = coefficient_df.loc[
    coefficient_df["selected"],
    "feature"
].tolist()

if len(selected_features) == 0:
    raise RuntimeError(
        "L1 selection retained zero features. "
        "Inspect the regularisation range."
    )

print(f"\nBest L1 inverse regularisation C: {best_c:.6g}")
print(f"Features selected by L1: {len(selected_features)}")

selected_feature_df = coefficient_df[
    coefficient_df["selected"]
].copy()

selected_feature_df["rank"] = np.arange(
    1,
    len(selected_feature_df) + 1
)

selected_feature_df = selected_feature_df[
    [
        "rank",
        "feature",
        "coefficient",
        "absolute_coefficient",
    ]
]

selected_feature_df.to_csv(
    SELECTED_FEATURES_FILE,
    index=False
)

coefficient_df.to_csv(
    COEFFICIENTS_FILE,
    index=False
)

if len(correlation_drops_df):
    correlation_drops_df.to_csv(
        CORRELATION_DROPS_FILE,
        index=False
    )
else:
    pd.DataFrame(
        columns=[
            "dropped_feature",
            "retained_correlated_feature",
            "absolute_spearman_rho",
        ]
    ).to_csv(
        CORRELATION_DROPS_FILE,
        index=False
    )

selected_indices = [
    correlation_retained_features.index(feature)
    for feature in selected_features
]

selected_scaled_column_names = [
    f"selected_scaled_{feature}"
    for feature in selected_features
]


def build_output_split(source_frame, scaled_matrix):
    metadata_columns = [
        c
        for c in [
            "full_image_id",
            "patient_id",
            "split",
            "binary_pathology",
            "n_lesions",
            "n_mass_lesions",
            "n_calcification_lesions",
            "radiomics_lesion_group",
            "preprocessed_path",
            "full_mammogram_path",
        ]
        if c in source_frame.columns
    ]

    metadata = source_frame[
        metadata_columns
    ].reset_index(drop=True)

    selected_matrix = scaled_matrix[:, selected_indices]

    selected_frame = pd.DataFrame(
        selected_matrix,
        columns=selected_scaled_column_names,
    )

    return pd.concat(
        [metadata, selected_frame],
        axis=1
    )


selected_train = build_output_split(train_df, X_train)
selected_validation = build_output_split(validation_df, X_validation)
selected_test = build_output_split(test_df, X_test)

selected_dataset = pd.concat(
    [
        selected_train,
        selected_validation,
        selected_test,
    ],
    ignore_index=True
)

selected_dataset.to_csv(
    SELECTED_DATA_FILE,
    index=False
)

save_correlation_heatmap(
    full_correlation_matrix,
    FIGURE_DIR / "training_correlation_all_candidate_features.png",
    "Training-Set Absolute Spearman Correlation Before Filtering",
)

selected_raw_correlation = train_df[
    selected_features
].corr(method=CORRELATION_METHOD).abs()

save_correlation_heatmap(
    selected_raw_correlation,
    FIGURE_DIR / "training_correlation_selected_features.png",
    "Training-Set Absolute Spearman Correlation of Selected Features",
)

figure_features = selected_feature_df.head(
    min(30, len(selected_feature_df))
).sort_values("coefficient")

fig = plt.figure(
    figsize=(
        9,
        max(5, 0.32 * len(figure_features))
    )
)

labels = [
    value.replace("original_", "")
        .replace("_mean", " mean")
        .replace("_max", " max")
    for value in figure_features["feature"]
]

plt.barh(
    labels,
    figure_features["coefficient"],
)

plt.xlabel("L1 Logistic Regression Coefficient")
plt.ylabel("Selected Radiomics Feature")
plt.title(
    f"Training-Only L1 Radiomics Feature Selection "
    f"(C={best_c:.4g})"
)
plt.grid(axis="x", alpha=0.25)
plt.tight_layout()

fig.savefig(
    FIGURE_DIR / "selected_radiomics_coefficients.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

summary = {
    "input_mammograms": int(len(df)),
    "split_counts": {
        str(k): int(v)
        for k, v in df["split"].value_counts().items()
    },
    "participant_overlap": overlaps,
    "initial_predictor_candidates": int(len(predictor_columns)),
    "training_constant_features_removed": int(len(constant_features)),
    "features_entering_correlation_filter": int(len(nonconstant_features)),
    "correlation_method": CORRELATION_METHOD,
    "correlation_threshold": float(CORRELATION_THRESHOLD),
    "correlation_filtered_features_removed": int(len(correlation_drops_df)),
    "features_after_correlation_filter": int(
        len(correlation_retained_features)
    ),
    "standardisation": {
        "method": "StandardScaler",
        "fit_split": "train_only",
    },
    "supervised_selector": {
        "method": "L1 logistic regression",
        "scoring": "roc_auc",
        "cv_folds": int(CV_FOLDS),
        "fit_split": "train_only",
        "best_C": best_c,
        "selected_feature_count": int(len(selected_features)),
    },
    "selected_features": selected_features,
}

with SUMMARY_FILE.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print("\n" + "=" * 72)
print("RADIOMICS FEATURE SELECTION COMPLETE")
print("=" * 72)

print(f"\nInitial predictor candidates: {len(predictor_columns)}")
print(
    f"Training-constant features removed: "
    f"{len(constant_features)}"
)
print(
    f"Correlation-filtered features removed: "
    f"{len(correlation_drops_df)}"
)
print(
    f"Features after correlation filtering: "
    f"{len(correlation_retained_features)}"
)
print(f"Best L1 C: {best_c:.6g}")
print(
    f"Final selected radiomics features: "
    f"{len(selected_features)}"
)

print("\nTop selected features:")
print(
    selected_feature_df.head(20).to_string(index=False)
)

print("\nSelected dataset split counts:")
print(selected_dataset["split"].value_counts())

print("\nSelected dataset class counts by split:")
print(
    pd.crosstab(
        selected_dataset["split"],
        selected_dataset["binary_pathology"]
    )
)

print("\nSaved selected dataset:")
print(SELECTED_DATA_FILE.resolve())

print("\nSaved selected-feature list:")
print(SELECTED_FEATURES_FILE.resolve())

print("\nSaved correlation-filter audit:")
print(CORRELATION_DROPS_FILE.resolve())

print("\nSaved L1 coefficient audit:")
print(COEFFICIENTS_FILE.resolve())

print("\nSaved scaler:")
print(SCALER_FILE.resolve())

print("\nSaved L1 selector:")
print(L1_SELECTOR_FILE.resolve())

print("\nSaved summary:")
print(SUMMARY_FILE.resolve())

print("\nSaved figures:")
print(FIGURE_DIR.resolve())
