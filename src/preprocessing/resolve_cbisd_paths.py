from pathlib import Path
import pandas as pd


# =========================================================
# Configuration
# =========================================================

CLASSIFIED_MANIFEST = "dicom_classified_manifest.csv"

CSV_FILES = [
    ("calc_train", "calc_case_description_train_set.csv"),
    ("calc_test", "calc_case_description_test_set.csv"),
    ("mass_train", "mass_case_description_train_set.csv"),
    ("mass_test", "mass_case_description_test_set.csv"),
]

OUTPUT_MANIFEST = "cbis_ddsm_resolved_manifest.csv"


# =========================================================
# Helper functions
# =========================================================

def extract_uids(path_value):
    """
    Extract StudyInstanceUID and SeriesInstanceUID from
    a CBIS-DDSM path.

    Expected structure:

        CaseName/
        StudyInstanceUID/
        SeriesInstanceUID/
        filename.dcm
    """

    if pd.isna(path_value):
        return None, None

    parts = str(path_value).replace("\\", "/").split("/")

    # Remove empty components
    parts = [p for p in parts if p]

    if len(parts) < 4:
        return None, None

    # The final components are:
    #
    # CaseName
    # StudyInstanceUID
    # SeriesInstanceUID
    # filename

    study_uid = parts[-3]
    series_uid = parts[-2]

    return study_uid, series_uid


# =========================================================
# Load DICOM manifest
# =========================================================

print("Loading DICOM manifest...")

dicom = pd.read_csv(CLASSIFIED_MANIFEST)

print(f"DICOM records: {len(dicom):,}")

# ---------------------------------------------------------
# Create UID lookup
# ---------------------------------------------------------

dicom["uid_key"] = (
    dicom["StudyInstanceUID"].astype(str)
    + "|"
    + dicom["SeriesInstanceUID"].astype(str)
)

print(
    "Unique Study/Series combinations:",
    dicom["uid_key"].nunique()
)


# =========================================================
# Process case-description CSVs
# =========================================================

all_records = []

for dataset_name, filename in CSV_FILES:

    print("\n" + "=" * 80)
    print(f"Processing: {filename}")
    print("=" * 80)

    df = pd.read_csv(filename)

    print(f"Rows: {len(df):,}")

    for index, row in df.iterrows():

        record = row.to_dict()

        record["dataset"] = dataset_name
        record["source_csv"] = filename
        record["source_row"] = index

        # -------------------------------------------------
        # Resolve three paths
        # -------------------------------------------------

        path_columns = {
            "image file path": "full_mammogram",
            "cropped image file path": "cropped",
            "ROI mask file path": "roi_mask",
        }

        for csv_column, role in path_columns.items():

            path_value = row.get(csv_column)

            study_uid, series_uid = extract_uids(
                path_value
            )

            record[f"{role}_study_uid"] = study_uid
            record[f"{role}_series_uid"] = series_uid

            # -------------------------------------------------
            # Find matching DICOM(s)
            # -------------------------------------------------

            if study_uid is None or series_uid is None:

                record[f"{role}_path"] = None
                record[f"{role}_filename"] = None

                continue

            key = f"{study_uid}|{series_uid}"

            matches = dicom[
                dicom["uid_key"] == key
            ]

            # -------------------------------------------------
            # Filter by image role
            # -------------------------------------------------

            if role == "full_mammogram":

                role_matches = matches[
                    matches["image_role"] == "full_mammogram"
                ]

                # If no explicitly identified full image,
                # look at unknown images.
                if len(role_matches) == 0:

                    role_matches = matches[
                        matches["image_role"] == "unknown"
                    ]

            else:

                role_matches = matches[
                    matches["image_role"] == role
                ]

            # -------------------------------------------------
            # Record result
            # -------------------------------------------------

            if len(role_matches) == 1:

                match = role_matches.iloc[0]

                record[f"{role}_path"] = (
                    match["dicom_path"]
                )

                record[f"{role}_filename"] = (
                    match["filename"]
                )

            elif len(role_matches) > 1:

                # Multiple possible files
                record[f"{role}_path"] = None
                record[f"{role}_filename"] = None

            else:

                record[f"{role}_path"] = None
                record[f"{role}_filename"] = None

        all_records.append(record)


# =========================================================
# Create final manifest
# =========================================================

resolved = pd.DataFrame(all_records)


# =========================================================
# Determine resolution status
# =========================================================

def resolution_status(row):

    paths = [
        row.get("full_mammogram_path"),
        row.get("cropped_path"),
        row.get("roi_mask_path"),
    ]

    found = sum(
        pd.notna(p)
        for p in paths
    )

    if found == 3:
        return "complete"

    elif found == 2:
        return "partial_2"

    elif found == 1:
        return "partial_1"

    else:
        return "unresolved"


resolved["resolution_status"] = (
    resolved.apply(
        resolution_status,
        axis=1
    )
)


# =========================================================
# Save
# =========================================================

resolved.to_csv(
    OUTPUT_MANIFEST,
    index=False
)


# =========================================================
# Summary
# =========================================================

print("\n")
print("=" * 80)
print("RESOLUTION SUMMARY")
print("=" * 80)

print(
    "\nResolution status:"
)

print(
    resolved["resolution_status"]
    .value_counts()
)


print(
    "\nFull mammograms resolved:",
    resolved["full_mammogram_path"]
    .notna()
    .sum(),
    "/",
    len(resolved)
)

print(
    "Cropped images resolved:",
    resolved["cropped_path"]
    .notna()
    .sum(),
    "/",
    len(resolved)
)

print(
    "ROI masks resolved:",
    resolved["roi_mask_path"]
    .notna()
    .sum(),
    "/",
    len(resolved)
)


print("\nDataset breakdown:")

print(
    resolved["dataset"]
    .value_counts()
)


print("\nSaved:")
print(OUTPUT_MANIFEST)