import pandas as pd


INPUT = (
    "../data/processed/cbis_ddsm/"
    "cbis_ddsm_processed_manifest.csv"
)

OUTPUT = (
    "../data/processed/cbis_ddsm/"
    "cbis_ddsm_full_image_manifest.csv"
)


df = pd.read_csv(INPUT)


# ============================================================
# Binary pathology mapping
# ============================================================

def binary_pathology(values):
    """
    Whole-mammogram label:

    If any abnormality on the mammogram is malignant,
    label the mammogram MALIGNANT.

    Otherwise label it BENIGN.
    """

    values = set(
        str(v).strip().upper()
        for v in values
        if pd.notna(v)
    )

    if "MALIGNANT" in values:
        return "MALIGNANT"

    return "BENIGN"


def pathology_list(values):
    return ",".join(
        sorted(
            set(
                str(v).strip()
                for v in values
                if pd.notna(v)
            )
        )
    )


def abnormality_types(values):
    return ",".join(
        sorted(
            set(
                str(v).strip()
                for v in values
                if pd.notna(v)
            )
        )
    )


# ============================================================
# One record per unique full mammogram
# ============================================================

records = []

for full_path, group in df.groupby(
    "full_mammogram_path",
    sort=False
):

    first = group.iloc[0]

    record = {
        "patient_id":
            first["patient_id"],

        "left_or_right_breast":
            first["left or right breast"],

        "image_view":
            first["image view"],

        "full_mammogram_path":
            full_path,

        # One of your generated JPEGs representing
        # this source mammogram.
        "full_jpeg_path":
            first["full_jpeg_path"],

        "binary_pathology":
            binary_pathology(
                group["pathology"]
            ),

        "original_pathologies":
            pathology_list(
                group["pathology"]
            ),

        "abnormality_types":
            abnormality_types(
                group["abnormality type"]
            ),

        "n_abnormalities":
            len(group),

        "source_datasets":
            ",".join(
                sorted(
                    set(group["dataset"])
                )
            ),

        "contains_mass":
            (
                group["abnormality type"]
                .astype(str)
                .str.lower()
                .eq("mass")
                .any()
            ),

        "contains_calcification":
            (
                group["abnormality type"]
                .astype(str)
                .str.lower()
                .eq("calcification")
                .any()
            ),
    }

    records.append(record)


images = pd.DataFrame(records)


# ============================================================
# Validation
# ============================================================

print("=" * 72)
print("FULL-IMAGE MANIFEST")
print("=" * 72)

print("\nRows:")
print(len(images))

print("\nUnique participants:")
print(images["patient_id"].nunique())

print("\nBinary pathology:")
print(
    images["binary_pathology"]
    .value_counts()
)

print("\nAbnormalities per mammogram:")
print(
    images["n_abnormalities"]
    .value_counts()
    .sort_index()
)

print("\nOriginal pathology combinations:")
print(
    images["original_pathologies"]
    .value_counts()
)

print("\nContains mass:")
print(
    images["contains_mass"]
    .value_counts()
)

print("\nContains calcification:")
print(
    images["contains_calcification"]
    .value_counts()
)


images.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)