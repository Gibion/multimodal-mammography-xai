from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd()

LESION_FEATURES_FILE = Path(
    "data/processed/cbis_ddsm/radiomics/"
    "cbis_ddsm_radiomics_lesion_features.csv"
)

FULL_IMAGE_MANIFEST_CANDIDATES = [
    Path(
        "data/processed/cbis_ddsm/manifests/"
        "cbis_ddsm_full_image_manifest_preprocessed_v4.csv"
    ),
    Path(
        "data/processed/cbis_ddsm/manifests/"
        "cbis_ddsm_full_image_manifest_split.csv"
    ),
    Path(
        "data/processed/cbis_ddsm/"
        "cbis_ddsm_full_image_manifest_split.csv"
    ),
]

OUTPUT_DIR = Path(
    "data/processed/cbis_ddsm/radiomics"
)

OUTPUT_FILE = OUTPUT_DIR / "cbis_ddsm_radiomics_mammogram_features.csv"
QA_FILE = OUTPUT_DIR / "cbis_ddsm_radiomics_mammogram_qa.csv"
SUMMARY_FILE = OUTPUT_DIR / "cbis_ddsm_radiomics_mammogram_summary.json"

FEATURE_PREFIX = "original_"


def find_existing_file(candidates):
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find a full-image manifest. Checked:\n"
        + "\n".join(str(path.resolve()) for path in candidates)
    )


def pathology_to_binary(value):
    text = str(value).strip().upper()
    if "MALIGNANT" in text:
        return "MALIGNANT"
    if text in {"BENIGN", "BENIGN_WITHOUT_CALLBACK"}:
        return "BENIGN"
    return np.nan


def combine_pathology(values):
    labels = [
        pathology_to_binary(v)
        for v in values
    ]
    labels = [v for v in labels if pd.notna(v)]
    if not labels:
        return np.nan
    return "MALIGNANT" if "MALIGNANT" in labels else "BENIGN"


def combine_lesion_types(values):
    types = {
        str(v).strip().lower()
        for v in values
        if pd.notna(v)
    }
    has_mass = "mass" in types
    has_calc = "calcification" in types
    if has_mass and has_calc:
        return "mass+calc"
    if has_mass:
        return "mass"
    if has_calc:
        return "calc"
    return "unknown"


def count_type(values, target):
    return sum(
        1
        for v in values
        if str(v).strip().lower() == target
    )


if not LESION_FEATURES_FILE.exists():
    raise FileNotFoundError(
        f"Lesion-level radiomics file not found:\n"
        f"{LESION_FEATURES_FILE.resolve()}"
    )

full_manifest_path = find_existing_file(
    FULL_IMAGE_MANIFEST_CANDIDATES
)

lesion_df = pd.read_csv(
    LESION_FEATURES_FILE
)

full_df = pd.read_csv(
    full_manifest_path
)

print("=" * 72)
print("AGGREGATE LESION-LEVEL RADIOMICS TO MAMMOGRAM LEVEL")
print("=" * 72)

print(f"\nLesion-level feature rows: {len(lesion_df):,}")
print(f"Full-image manifest rows: {len(full_df):,}")
print("\nFull-image manifest:")
print(full_manifest_path.resolve())

required_lesion_columns = [
    "full_image_id",
    "patient_id",
    "dataset",
    "abnormality_id",
    "abnormality_type",
    "pathology",
]

missing_lesion_columns = [
    c for c in required_lesion_columns
    if c not in lesion_df.columns
]

if missing_lesion_columns:
    raise ValueError(
        "Lesion-level radiomics file is missing required columns: "
        + ", ".join(missing_lesion_columns)
    )

feature_columns = [
    c for c in lesion_df.columns
    if c.startswith(FEATURE_PREFIX)
]

if not feature_columns:
    raise ValueError(
        f"No radiomics feature columns beginning with '{FEATURE_PREFIX}' found."
    )

for c in feature_columns:
    lesion_df[c] = pd.to_numeric(
        lesion_df[c],
        errors="coerce"
    )

if "image_id" not in full_df.columns:
    raise ValueError(
        "Full-image manifest must contain 'image_id'."
    )

if full_df["image_id"].duplicated().any():
    raise ValueError(
        "Full-image manifest contains duplicate image_id values."
    )

full_lookup_columns = [
    c
    for c in [
        "image_id",
        "patient_id",
        "split",
        "binary_pathology",
        "laterality",
        "image_view",
        "left or right breast",
        "image view",
        "contains_mass",
        "contains_calcification",
        "preprocessed_path",
        "full_mammogram_path",
    ]
    if c in full_df.columns
]

full_lookup = (
    full_df[full_lookup_columns]
    .copy()
    .rename(columns={"image_id": "full_image_id"})
)

grouped_records = []

for full_image_id, group in lesion_df.groupby(
    "full_image_id",
    sort=False
):
    record = {
        "full_image_id": full_image_id,
        "patient_id_radiomics": group["patient_id"].iloc[0],
        "n_lesions": int(len(group)),
        "n_mass_lesions": int(
            count_type(group["abnormality_type"], "mass")
        ),
        "n_calcification_lesions": int(
            count_type(group["abnormality_type"], "calcification")
        ),
        "radiomics_lesion_group": combine_lesion_types(
            group["abnormality_type"]
        ),
        "radiomics_binary_pathology": combine_pathology(
            group["pathology"]
        ),
        "radiomics_source_datasets": ",".join(
            sorted(
                set(
                    group["dataset"]
                    .dropna()
                    .astype(str)
                )
            )
        ),
    }

    for feature in feature_columns:
        values = pd.to_numeric(
            group[feature],
            errors="coerce"
        )

        record[f"{feature}_mean"] = float(values.mean())
        record[f"{feature}_max"] = float(values.max())

    grouped_records.append(record)

agg_df = pd.DataFrame(grouped_records)

print(
    f"\nUnique mammograms represented by radiomics: "
    f"{len(agg_df):,}"
)

merged = agg_df.merge(
    full_lookup,
    on="full_image_id",
    how="left",
    validate="one_to_one",
    indicator=True
)

print("\nFull-image manifest matching:")
print(merged["_merge"].value_counts())

merged = merged.drop(columns="_merge")

if "patient_id" in merged.columns:
    merged["patient_id_match"] = (
        merged["patient_id_radiomics"].astype(str)
        ==
        merged["patient_id"].astype(str)
    )
else:
    merged["patient_id"] = merged["patient_id_radiomics"]
    merged["patient_id_match"] = True

if "binary_pathology" in merged.columns:
    merged["pathology_match"] = (
        merged["radiomics_binary_pathology"].astype(str)
        ==
        merged["binary_pathology"].astype(str)
    )
else:
    merged["binary_pathology"] = merged["radiomics_binary_pathology"]
    merged["pathology_match"] = True

aggregated_feature_columns = [
    c
    for c in merged.columns
    if (
        c.startswith(FEATURE_PREFIX)
        and (
            c.endswith("_mean")
            or c.endswith("_max")
        )
    )
]

predictor_columns = (
    aggregated_feature_columns
    + ["n_lesions"]
)

qa_records = []

for column in predictor_columns:
    series = pd.to_numeric(
        merged[column],
        errors="coerce"
    )

    arr = series.to_numpy(dtype=float)

    finite = arr[np.isfinite(arr)]

    qa_records.append({
        "feature": column,
        "n_missing": int(np.isnan(arr).sum()),
        "n_infinite": int(np.isinf(arr).sum()),
        "n_unique_finite": int(pd.Series(finite).nunique()),
        "is_constant": bool(
            pd.Series(finite).nunique() <= 1
        ),
        "mean": float(np.mean(finite)) if len(finite) else np.nan,
        "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
        "min": float(np.min(finite)) if len(finite) else np.nan,
        "max": float(np.max(finite)) if len(finite) else np.nan,
    })

qa_df = pd.DataFrame(qa_records)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

merged.to_csv(
    OUTPUT_FILE,
    index=False
)

qa_df.to_csv(
    QA_FILE,
    index=False
)

n_constant = int(qa_df["is_constant"].sum())
n_features_with_missing = int((qa_df["n_missing"] > 0).sum())
n_features_with_inf = int((qa_df["n_infinite"] > 0).sum())

summary = {
    "lesion_level_rows": int(len(lesion_df)),
    "unique_radiomics_mammograms": int(len(merged)),
    "original_radiomics_features_per_lesion": int(len(feature_columns)),
    "aggregated_radiomics_features": int(len(aggregated_feature_columns)),
    "additional_lesion_count_features": 1,
    "total_radiomics_predictor_candidates": int(len(predictor_columns)),
    "constant_predictor_candidates": n_constant,
    "predictor_candidates_with_missing_values": n_features_with_missing,
    "predictor_candidates_with_infinite_values": n_features_with_inf,
    "patient_id_mismatches": int((~merged["patient_id_match"]).sum()),
    "pathology_mismatches": int((~merged["pathology_match"]).sum()),
}

if "split" in merged.columns:
    summary["split_counts"] = {
        str(k): int(v)
        for k, v in merged["split"].value_counts().items()
    }

with SUMMARY_FILE.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print("\n")
print("=" * 72)
print("MAMMOGRAM-LEVEL RADIOMICS SUMMARY")
print("=" * 72)

print(f"\nLesion-level rows: {len(lesion_df):,}")
print(f"Unique mammograms represented: {len(merged):,}")

print("\nLesions per mammogram:")
print(
    merged["n_lesions"]
    .value_counts()
    .sort_index()
)

print(
    f"\nOriginal features per lesion: "
    f"{len(feature_columns):,}"
)

print(
    f"Aggregated mean/max features: "
    f"{len(aggregated_feature_columns):,}"
)

print("Additional lesion-count predictor: 1")
print(
    f"Total predictor candidates: "
    f"{len(predictor_columns):,}"
)

if "split" in merged.columns:
    print("\nSplit counts:")
    print(merged["split"].value_counts())

if (
    "binary_pathology" in merged.columns
    and
    "split" in merged.columns
):
    print("\nClass counts by split:")
    print(
        pd.crosstab(
            merged["split"],
            merged["binary_pathology"]
        )
    )

print("\nLesion group:")
print(
    merged["radiomics_lesion_group"]
    .value_counts()
)

print("\nPatient ID agreement:")
print(
    merged["patient_id_match"]
    .value_counts()
)

print("\nPathology agreement:")
print(
    merged["pathology_match"]
    .value_counts()
)

print(
    f"\nFeatures containing missing values: "
    f"{n_features_with_missing}"
)

print(
    f"Features containing infinite values: "
    f"{n_features_with_inf}"
)

print(
    f"Constant predictor candidates: "
    f"{n_constant}"
)

if n_constant:
    print("\nConstant features:")
    print(
        qa_df.loc[
            qa_df["is_constant"],
            "feature"
        ].to_string(index=False)
    )

print("\nSaved mammogram-level radiomics file:")
print(OUTPUT_FILE.resolve())

print("\nSaved feature QA file:")
print(QA_FILE.resolve())

print("\nSaved summary:")
print(SUMMARY_FILE.resolve())
