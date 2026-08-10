import pandas as pd


# ============================================================
# Configuration
# ============================================================

CLASSIFIED_MANIFEST = "dicom_classified_manifest.csv"

CSV_FILES = [
    ("calc_train", "calc_case_description_train_set.csv"),
    ("calc_test", "calc_case_description_test_set.csv"),
    ("mass_train", "mass_case_description_train_set.csv"),
    ("mass_test", "mass_case_description_test_set.csv"),
]

OUTPUT_FILE = "cbis_ddsm_resolved_manifest_v2.csv"


# ============================================================
# Helpers
# ============================================================

def extract_uids(path_value):
    """
    Extract StudyInstanceUID and SeriesInstanceUID from a
    CBIS-DDSM CSV path.

    Expected general form:

        CaseName/
        StudyInstanceUID/
        SeriesInstanceUID/
        filename.dcm
    """

    if pd.isna(path_value):
        return None, None

    value = str(path_value).strip()

    # Normalize separators
    value = value.replace("\\", "/")

    parts = [
        part.strip()
        for part in value.split("/")
        if part.strip()
    ]

    if len(parts) < 4:
        return None, None

    study_uid = parts[-3]
    series_uid = parts[-2]

    return study_uid, series_uid


def uid_key(study_uid, series_uid):
    if study_uid is None or series_uid is None:
        return None

    return f"{study_uid}|{series_uid}"


def get_matches(dicom, study_uid, series_uid):
    """
    Return all DICOM files belonging to this Study/Series UID.
    """

    key = uid_key(study_uid, series_uid)

    if key is None:
        return dicom.iloc[0:0]

    return dicom[dicom["uid_key"] == key]


def deduplicate_matches(matches):
    """
    Remove the same physical DICOM if it was discovered from
    both CSV path columns.
    """

    if len(matches) == 0:
        return matches

    return matches.drop_duplicates(
        subset=["dicom_path"]
    )


def choose_single_role(matches, role):
    """
    Return the single DICOM having the requested actual image role.

    Returns:
        match, status

    status:
        resolved
        missing
        ambiguous
    """

    role_matches = matches[
        matches["image_role"] == role
    ]

    role_matches = deduplicate_matches(role_matches)

    if len(role_matches) == 1:
        return role_matches.iloc[0], "resolved"

    if len(role_matches) == 0:
        return None, "missing"

    return None, "ambiguous"


# ============================================================
# Load actual DICOM manifest
# ============================================================

print("Loading classified DICOM manifest...")

dicom = pd.read_csv(CLASSIFIED_MANIFEST)

dicom["uid_key"] = (
    dicom["StudyInstanceUID"].astype(str)
    + "|"
    + dicom["SeriesInstanceUID"].astype(str)
)

print(f"DICOM files: {len(dicom):,}")
print(
    f"Study/Series combinations: "
    f"{dicom['uid_key'].nunique():,}"
)


# ============================================================
# Process all four description CSVs
# ============================================================

records = []


for dataset_name, csv_file in CSV_FILES:

    print("\n" + "=" * 72)
    print(csv_file)
    print("=" * 72)

    cases = pd.read_csv(csv_file)

    print(f"Rows: {len(cases):,}")

    for source_row, row in cases.iterrows():

        record = row.to_dict()

        record["dataset"] = dataset_name
        record["source_csv"] = csv_file
        record["source_row"] = source_row

        # ====================================================
        # Extract all original UID references
        # ====================================================

        full_study, full_series = extract_uids(
            row["image file path"]
        )

        crop_study, crop_series = extract_uids(
            row["cropped image file path"]
        )

        roi_study, roi_series = extract_uids(
            row["ROI mask file path"]
        )

        record["full_csv_study_uid"] = full_study
        record["full_csv_series_uid"] = full_series

        record["crop_csv_study_uid"] = crop_study
        record["crop_csv_series_uid"] = crop_series

        record["roi_csv_study_uid"] = roi_study
        record["roi_csv_series_uid"] = roi_series


        # ====================================================
        # FULL MAMMOGRAM
        # ====================================================

        full_candidates = get_matches(
            dicom,
            full_study,
            full_series
        )

        full_match, full_status = choose_single_role(
            full_candidates,
            "full_mammogram"
        )

        # Some genuine full mammograms have blank
        # SeriesDescription and were classified "unknown".
        if full_match is None:

            unknown_match, unknown_status = choose_single_role(
                full_candidates,
                "unknown"
            )

            if unknown_match is not None:
                full_match = unknown_match
                full_status = "resolved_unknown_description"


        if full_match is not None:

            record["full_mammogram_path"] = (
                full_match["dicom_path"]
            )

            record["full_mammogram_filename"] = (
                full_match["filename"]
            )

            record["full_mammogram_role"] = (
                full_match["image_role"]
            )

        else:

            record["full_mammogram_path"] = None
            record["full_mammogram_filename"] = None
            record["full_mammogram_role"] = None

        record["full_match_status"] = full_status


        # ====================================================
        # CROP + ROI
        #
        # IMPORTANT:
        #
        # Do NOT assume the cropped CSV column contains a crop
        # and the ROI CSV column contains a mask.
        #
        # Both referenced UID groups are candidate sources.
        # The DICOM's actual image_role determines its meaning.
        # ====================================================

        crop_candidates = get_matches(
            dicom,
            crop_study,
            crop_series
        )

        roi_candidates = get_matches(
            dicom,
            roi_study,
            roi_series
        )

        lesion_candidates = pd.concat(
            [
                crop_candidates,
                roi_candidates
            ],
            ignore_index=True
        )

        lesion_candidates = deduplicate_matches(
            lesion_candidates
        )


        # ----------------------------------------------------
        # Actual cropped image
        # ----------------------------------------------------

        cropped_match, cropped_status = choose_single_role(
            lesion_candidates,
            "cropped"
        )

        if cropped_match is not None:

            record["cropped_path"] = (
                cropped_match["dicom_path"]
            )

            record["cropped_filename"] = (
                cropped_match["filename"]
            )

        else:

            record["cropped_path"] = None
            record["cropped_filename"] = None

        record["cropped_match_status"] = cropped_status


        # ----------------------------------------------------
        # Actual ROI mask
        # ----------------------------------------------------

        roi_match, roi_status = choose_single_role(
            lesion_candidates,
            "roi_mask"
        )

        if roi_match is not None:

            record["roi_mask_path"] = (
                roi_match["dicom_path"]
            )

            record["roi_mask_filename"] = (
                roi_match["filename"]
            )

        else:

            record["roi_mask_path"] = None
            record["roi_mask_filename"] = None

        record["roi_match_status"] = roi_status


        # ====================================================
        # Detect CSV path reversal
        # ====================================================

        direct_crop_matches = get_matches(
            dicom,
            crop_study,
            crop_series
        )

        direct_roi_matches = get_matches(
            dicom,
            roi_study,
            roi_series
        )

        crop_csv_roles = set(
            direct_crop_matches[
                "image_role"
            ].dropna()
        )

        roi_csv_roles = set(
            direct_roi_matches[
                "image_role"
            ].dropna()
        )

        record["csv_paths_reversed"] = (
            crop_csv_roles == {"roi_mask"}
            and
            roi_csv_roles == {"cropped"}
        )


        # ====================================================
        # Availability flags
        # ====================================================

        record["has_full_mammogram"] = (
            pd.notna(
                record["full_mammogram_path"]
            )
        )

        record["has_cropped"] = (
            pd.notna(
                record["cropped_path"]
            )
        )

        record["has_roi_mask"] = (
            pd.notna(
                record["roi_mask_path"]
            )
        )


        # ====================================================
        # Overall resolution
        # ====================================================

        found = sum([
            record["has_full_mammogram"],
            record["has_cropped"],
            record["has_roi_mask"]
        ])

        if found == 3:
            record["resolution_status"] = "complete"

        elif found == 2:
            record["resolution_status"] = "partial_2"

        elif found == 1:
            record["resolution_status"] = "partial_1"

        else:
            record["resolution_status"] = "unresolved"


        records.append(record)


# ============================================================
# Final dataframe
# ============================================================

resolved = pd.DataFrame(records)


# ============================================================
# Save
# ============================================================

resolved.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Validation summary
# ============================================================

print("\n")
print("=" * 72)
print("FINAL RESOLUTION SUMMARY")
print("=" * 72)

print("\nTotal case-description rows:")
print(f"{len(resolved):,}")

print("\nResolution status:")
print(
    resolved["resolution_status"]
    .value_counts(dropna=False)
)

print("\nImage availability:")
print(
    "Full mammograms:",
    resolved["has_full_mammogram"].sum(),
    "/",
    len(resolved)
)

print(
    "Cropped images:",
    resolved["has_cropped"].sum(),
    "/",
    len(resolved)
)

print(
    "ROI masks:",
    resolved["has_roi_mask"].sum(),
    "/",
    len(resolved)
)

print("\nCSV paths detected as reversed:")
print(
    resolved["csv_paths_reversed"].sum()
)

print("\nReversed rows by dataset:")
print(
    resolved[
        resolved["csv_paths_reversed"]
    ]["dataset"].value_counts()
)

print("\nMissing ROI masks by dataset:")
print(
    resolved[
        ~resolved["has_roi_mask"]
    ]["dataset"].value_counts()
)

print("\nMissing cropped images by dataset:")
print(
    resolved[
        ~resolved["has_cropped"]
    ]["dataset"].value_counts()
)

print("\nMissing full mammograms by dataset:")
print(
    resolved[
        ~resolved["has_full_mammogram"]
    ]["dataset"].value_counts()
)

print("\nAmbiguous full matches:")
print(
    (
        resolved["full_match_status"]
        == "ambiguous"
    ).sum()
)

print("Ambiguous cropped matches:")
print(
    (
        resolved["cropped_match_status"]
        == "ambiguous"
    ).sum()
)

print("Ambiguous ROI matches:")
print(
    (
        resolved["roi_match_status"]
        == "ambiguous"
    ).sum()
)

print("\nSaved:")
print(OUTPUT_FILE)