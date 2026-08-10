from pathlib import Path
import pandas as pd
import pydicom
from tqdm import tqdm


MANIFEST_FILE = "dicom_manifest.csv"
OUTPUT_FILE = "dicom_classified_manifest.csv"


manifest = pd.read_csv(MANIFEST_FILE)

records = []


for _, row in tqdm(
    manifest.iterrows(),
    total=len(manifest),
    desc="Classifying DICOM files"
):

    path = Path(row["dicom_path"])

    try:

        ds = pydicom.dcmread(
            path,
            stop_before_pixels=True
        )

        rows = getattr(ds, "Rows", None)
        columns = getattr(ds, "Columns", None)

        series_description = str(
            getattr(ds, "SeriesDescription", "")
        ).strip().lower()

        patient_id = str(
            getattr(ds, "PatientID", "")
        )

        # -------------------------------------------------
        # Determine image role
        # -------------------------------------------------

        if "cropped images" in series_description:

            image_role = "cropped"

        elif "roi mask images" in series_description:

            image_role = "roi_mask"

        elif "full mammogram images" in series_description:

            image_role = "full_mammogram"

        elif series_description in ("", "none", "nan"):

            image_role = "unknown"

        else:

            image_role = "other"

        # -------------------------------------------------
        # Record information
        # -------------------------------------------------

        records.append({

            "metadata_index":
                row["metadata_index"],

            "PatientID_metadata":
                row["PatientID"],

            "PatientID_dicom":
                patient_id,

            "StudyInstanceUID":
                getattr(
                    ds,
                    "StudyInstanceUID",
                    None
                ),

            "SeriesInstanceUID":
                getattr(
                    ds,
                    "SeriesInstanceUID",
                    None
                ),

            "SOPInstanceUID":
                getattr(
                    ds,
                    "SOPInstanceUID",
                    None
                ),

            "Modality":
                getattr(
                    ds,
                    "Modality",
                    None
                ),

            "SeriesDescription":
                getattr(
                    ds,
                    "SeriesDescription",
                    None
                ),

            "Rows":
                rows,

            "Columns":
                columns,

            "BitsAllocated":
                getattr(
                    ds,
                    "BitsAllocated",
                    None
                ),

            "BitsStored":
                getattr(
                    ds,
                    "BitsStored",
                    None
                ),

            "PhotometricInterpretation":
                getattr(
                    ds,
                    "PhotometricInterpretation",
                    None
                ),

            "Laterality":
                getattr(
                    ds,
                    "Laterality",
                    None
                ),

            "PatientOrientation":
                getattr(
                    ds,
                    "PatientOrientation",
                    None
                ),

            "image_role":
                image_role,

            "filename":
                path.name,

            "dicom_path":
                str(path),

            "S5cmdManifestPath":
                row["S5cmdManifestPath"]
        })

    except Exception as e:

        print(
            f"\nCould not read: {path}"
        )
        print(e)


classified = pd.DataFrame(records)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

classified.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\nImage-role summary")
print("==================")

print(
    classified["image_role"]
    .value_counts(dropna=False)
)

print("\nSeries descriptions")
print("====================")

print(
    classified["SeriesDescription"]
    .value_counts(dropna=False)
)

print("\nSaved:")
print(OUTPUT_FILE)