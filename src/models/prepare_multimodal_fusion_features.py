from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path.cwd()

CNN_FEATURES_FILE = Path(
    "data/processed/cbis_ddsm/fusion/"
    "cbis_ddsm_resnet50v2_dense256_features.csv"
)

RADIOMICS_FEATURES_FILE = Path(
    "data/processed/cbis_ddsm/radiomics/feature_selection/"
    "cbis_ddsm_radiomics_selected_features.csv"
)

OUTPUT_DIR = Path(
    "data/processed/cbis_ddsm/fusion"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "cbis_ddsm_multimodal_fusion_features.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "cbis_ddsm_multimodal_fusion_features_summary.json"
)

CNN_SCALER_FILE = (
    OUTPUT_DIR
    / "cnn_feature_standard_scaler.joblib"
)

CNN_DROPPED_CONSTANTS_FILE = (
    OUTPUT_DIR
    / "cnn_training_constant_features.csv"
)

# Prefixes created by earlier pipeline stages.
CNN_PREFIX = "cnn_dense256_"
RADIOMICS_PREFIX = "selected_scaled_"


# ============================================================
# Helpers
# ============================================================

def validate_unique_ids(frame, name):
    if "full_image_id" not in frame.columns:
        raise ValueError(
            f"{name} is missing full_image_id."
        )

    if frame[
        "full_image_id"
    ].duplicated().any():
        duplicates = frame.loc[
            frame[
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
            f"{name} contains duplicate full_image_id rows:\n"
            + duplicates.head(
                20
            ).to_string(
                index=False
            )
        )


def compare_required_metadata(
    merged,
    left_suffix,
    right_suffix,
):
    comparisons = {}

    for base_name in [
        "patient_id",
        "split",
        "binary_pathology",
    ]:
        left = f"{base_name}_{left_suffix}"
        right = f"{base_name}_{right_suffix}"

        if (
            left not in merged.columns
            or right not in merged.columns
        ):
            raise ValueError(
                f"Expected metadata columns "
                f"'{left}' and '{right}'."
            )

        matches = (
            merged[
                left
            ].astype(
                str
            )
            ==
            merged[
                right
            ].astype(
                str
            )
        )

        comparisons[
            base_name
        ] = int(
            (
                ~matches
            ).sum()
        )

        if not matches.all():
            bad = merged.loc[
                ~matches,
                [
                    "full_image_id",
                    left,
                    right,
                ],
            ]

            raise ValueError(
                f"{base_name} mismatch found:\n"
                + bad.head(
                    20
                ).to_string(
                    index=False
                )
            )

    return comparisons


def find_constant_training_features(
    train_frame,
    columns,
):
    constant = []
    retained = []

    for column in columns:
        n_unique = (
            train_frame[
                column
            ]
            .nunique(
                dropna=True
            )
        )

        if n_unique <= 1:
            constant.append(
                column
            )
        else:
            retained.append(
                column
            )

    return (
        retained,
        constant,
    )


# ============================================================
# Validate input files
# ============================================================

if not CNN_FEATURES_FILE.exists():
    raise FileNotFoundError(
        f"CNN feature file not found:\n"
        f"{CNN_FEATURES_FILE.resolve()}"
    )

if not RADIOMICS_FEATURES_FILE.exists():
    raise FileNotFoundError(
        f"Radiomics feature file not found:\n"
        f"{RADIOMICS_FEATURES_FILE.resolve()}"
    )


# ============================================================
# Load feature tables
# ============================================================

cnn_df = pd.read_csv(
    CNN_FEATURES_FILE
)

rad_df = pd.read_csv(
    RADIOMICS_FEATURES_FILE
)

validate_unique_ids(
    cnn_df,
    "CNN feature table",
)

validate_unique_ids(
    rad_df,
    "Radiomics feature table",
)

cnn_feature_columns = [
    column
    for column in cnn_df.columns
    if column.startswith(
        CNN_PREFIX
    )
]

radiomics_feature_columns = [
    column
    for column in rad_df.columns
    if column.startswith(
        RADIOMICS_PREFIX
    )
]

if not cnn_feature_columns:
    raise ValueError(
        "No CNN dense_256 feature columns were found."
    )

if not radiomics_feature_columns:
    raise ValueError(
        "No selected radiomics feature columns were found."
    )


# ============================================================
# Join one-to-one on full_image_id
# ============================================================

cnn_keep = [
    column
    for column in [
        "full_image_id",
        "patient_id",
        "split",
        "binary_pathology",
        "preprocessed_path",
        "full_mammogram_path",
    ]
    if column in cnn_df.columns
] + cnn_feature_columns

rad_keep = [
    column
    for column in [
        "full_image_id",
        "patient_id",
        "split",
        "binary_pathology",
        "n_lesions",
        "n_mass_lesions",
        "n_calcification_lesions",
        "radiomics_lesion_group",
    ]
    if column in rad_df.columns
] + radiomics_feature_columns

merged = cnn_df[
    cnn_keep
].merge(
    rad_df[
        rad_keep
    ],
    on="full_image_id",
    how="inner",
    suffixes=(
        "_cnn",
        "_radiomics",
    ),
    validate="one_to_one",
    indicator=True,
)

print("=" * 72)
print("PREPARE MULTIMODAL FUSION FEATURES")
print("=" * 72)

print(
    f"\nCNN rows: "
    f"{len(cnn_df):,}"
)

print(
    f"Radiomics rows: "
    f"{len(rad_df):,}"
)

print(
    f"Matched rows: "
    f"{len(merged):,}"
)

if len(
    merged
) != len(
    rad_df
):
    raise ValueError(
        "Matched fusion cohort size does not equal "
        "radiomics cohort size."
    )

metadata_mismatches = compare_required_metadata(
    merged,
    "cnn",
    "radiomics",
)

merged = merged.drop(
    columns=[
        "_merge"
    ]
)


# ============================================================
# Build clean metadata columns
# ============================================================

clean = pd.DataFrame({
    "full_image_id": merged[
        "full_image_id"
    ],
    "patient_id": merged[
        "patient_id_cnn"
    ],
    "split": merged[
        "split_cnn"
    ],
    "binary_pathology": merged[
        "binary_pathology_cnn"
    ],
})

for source_column, output_column in [
    (
        "preprocessed_path",
        "preprocessed_path",
    ),
    (
        "full_mammogram_path",
        "full_mammogram_path",
    ),
    (
        "n_lesions",
        "n_lesions",
    ),
    (
        "n_mass_lesions",
        "n_mass_lesions",
    ),
    (
        "n_calcification_lesions",
        "n_calcification_lesions",
    ),
    (
        "radiomics_lesion_group",
        "radiomics_lesion_group",
    ),
]:
    if source_column in merged.columns:
        clean[
            output_column
        ] = merged[
            source_column
        ]


# ============================================================
# Numeric QA
# ============================================================

for column in (
    cnn_feature_columns
    + radiomics_feature_columns
):
    merged[
        column
    ] = pd.to_numeric(
        merged[
            column
        ],
        errors="coerce"
    )

all_numeric = merged[
    cnn_feature_columns
    + radiomics_feature_columns
].to_numpy(
    dtype=float
)

if np.isnan(
    all_numeric
).any():
    raise ValueError(
        "NaN values found before fusion preparation."
    )

if np.isinf(
    all_numeric
).any():
    raise ValueError(
        "Infinite values found before fusion preparation."
    )


# ============================================================
# Training-only constant CNN feature removal
# ============================================================

train_mask = (
    merged[
        "split_cnn"
    ] == "train"
)

train_merged = merged.loc[
    train_mask
].copy()

(
    retained_cnn_features,
    constant_cnn_features,
) = find_constant_training_features(
    train_merged,
    cnn_feature_columns,
)

print(
    f"\nOriginal CNN feature dimension: "
    f"{len(cnn_feature_columns)}"
)

print(
    f"Training-constant CNN features removed: "
    f"{len(constant_cnn_features)}"
)

print(
    f"CNN features retained: "
    f"{len(retained_cnn_features)}"
)

CNN_DROPPED_CONSTANTS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

pd.DataFrame({
    "constant_cnn_feature": (
        constant_cnn_features
    )
}).to_csv(
    CNN_DROPPED_CONSTANTS_FILE,
    index=False,
)


# ============================================================
# Training-only standardisation of retained CNN features
# ============================================================

scaler = StandardScaler()

X_train_cnn = train_merged[
    retained_cnn_features
].to_numpy(
    dtype=float
)

scaler.fit(
    X_train_cnn
)

joblib.dump(
    scaler,
    CNN_SCALER_FILE,
)

X_all_cnn_scaled = scaler.transform(
    merged[
        retained_cnn_features
    ].to_numpy(
        dtype=float
    )
)

scaled_cnn_columns = [
    f"cnn_scaled_{index:03d}"
    for index in range(
        len(
            retained_cnn_features
        )
    )
]

scaled_cnn_df = pd.DataFrame(
    X_all_cnn_scaled,
    columns=scaled_cnn_columns,
)

# The radiomics selected features are already standardized using
# training-only statistics in select_radiomics_features.py.
radiomics_df = merged[
    radiomics_feature_columns
].reset_index(
    drop=True
).copy()

radiomics_output_columns = [
    f"radiomics_selected_{index:02d}"
    for index in range(
        len(
            radiomics_feature_columns
        )
    )
]

radiomics_df.columns = (
    radiomics_output_columns
)


# ============================================================
# Concatenate multimodal representation
# ============================================================

fusion_df = pd.concat(
    [
        clean.reset_index(
            drop=True
        ),
        scaled_cnn_df,
        radiomics_df,
    ],
    axis=1,
)

fusion_feature_columns = (
    scaled_cnn_columns
    + radiomics_output_columns
)

fusion_array = fusion_df[
    fusion_feature_columns
].to_numpy(
    dtype=float
)

n_nan = int(
    np.isnan(
        fusion_array
    ).sum()
)

n_inf = int(
    np.isinf(
        fusion_array
    ).sum()
)

if n_nan > 0:
    raise ValueError(
        f"Fusion matrix contains {n_nan} NaN values."
    )

if n_inf > 0:
    raise ValueError(
        f"Fusion matrix contains {n_inf} infinite values."
    )

if fusion_df[
    "full_image_id"
].duplicated().any():
    raise ValueError(
        "Duplicate full_image_id rows found in fusion dataset."
    )


# ============================================================
# Save prepared fusion dataset
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

fusion_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# Summary
# ============================================================

summary = {
    "matched_mammograms": int(
        len(
            fusion_df
        )
    ),
    "split_counts": {
        str(
            key
        ): int(
            value
        )
        for key, value
        in fusion_df[
            "split"
        ].value_counts().items()
    },
    "class_counts_by_split": {
        split_name: {
            str(
                key
            ): int(
                value
            )
            for key, value
            in fusion_df.loc[
                fusion_df[
                    "split"
                ] == split_name,
                "binary_pathology",
            ].value_counts().items()
        }
        for split_name in [
            "train",
            "validation",
            "test",
        ]
    },
    "metadata_mismatches": (
        metadata_mismatches
    ),
    "cnn_original_dimension": int(
        len(
            cnn_feature_columns
        )
    ),
    "cnn_training_constant_features_removed": int(
        len(
            constant_cnn_features
        )
    ),
    "cnn_retained_dimension": int(
        len(
            retained_cnn_features
        )
    ),
    "radiomics_dimension": int(
        len(
            radiomics_feature_columns
        )
    ),
    "fusion_dimension": int(
        len(
            fusion_feature_columns
        )
    ),
    "cnn_scaling": {
        "method": "StandardScaler",
        "fit_split": "train_only",
    },
    "radiomics_scaling": {
        "status": "already standardised",
        "source": (
            "select_radiomics_features.py"
        ),
        "fit_split": "train_only",
    },
    "nan_values": n_nan,
    "infinite_values": n_inf,
    "constant_cnn_features": (
        constant_cnn_features
    ),
    "retained_cnn_source_features": (
        retained_cnn_features
    ),
    "radiomics_source_features": (
        radiomics_feature_columns
    ),
}

with SUMMARY_FILE.open(
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
print("MULTIMODAL FUSION FEATURE PREPARATION COMPLETE")
print("=" * 72)

print(
    f"\nMatched mammograms: "
    f"{len(fusion_df):,}"
)

print(
    "\nSplit counts:"
)

print(
    fusion_df[
        "split"
    ].value_counts()
)

print(
    "\nClass counts by split:"
)

print(
    pd.crosstab(
        fusion_df[
            "split"
        ],
        fusion_df[
            "binary_pathology"
        ],
    )
)

print(
    f"\nOriginal CNN dimension: "
    f"{len(cnn_feature_columns)}"
)

print(
    f"Training-constant CNN features removed: "
    f"{len(constant_cnn_features)}"
)

print(
    f"Retained CNN dimension: "
    f"{len(retained_cnn_features)}"
)

print(
    f"Radiomics dimension: "
    f"{len(radiomics_feature_columns)}"
)

print(
    f"Final fusion dimension: "
    f"{len(fusion_feature_columns)}"
)

print(
    f"\nNaN values: "
    f"{n_nan}"
)

print(
    f"Infinite values: "
    f"{n_inf}"
)

print(
    "\nSaved prepared fusion dataset:"
)

print(
    OUTPUT_FILE.resolve()
)

print(
    "\nSaved CNN scaler:"
)

print(
    CNN_SCALER_FILE.resolve()
)

print(
    "\nSaved constant-feature audit:"
)

print(
    CNN_DROPPED_CONSTANTS_FILE.resolve()
)

print(
    "\nSaved summary:"
)

print(
    SUMMARY_FILE.resolve()
)
