from pathlib import Path
import pandas as pd
import pydicom
from tqdm import tqdm

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MANIFEST_FILE = "dicom_manifest.csv"
OUTPUT_FILE = "dicom_header_manifest.csv"

# ---------------------------------------------------------
# Load manifest
# ---------------------------------------------------------

manifest = pd.read_csv(MANIFEST_FILE)

print(f"DICOM records: {len(manifest):,}")
print()

# ---------------------------------------------------------
# Read DICOM headers
# ---------------------------------------------------------

records = []

for _, row in tqdm(
    manifest.iterrows(),
    total=len(manifest),
    desc="Reading DICOM headers"
):

    dcm_path = Path(row["dicom_path"])

    try:

        ds = pydicom.dcmread(
            dcm_path,
            stop_before_pixels=True
        )

        records.append({
            "metadata_index": row["metadata_index"],

            "PatientID": row["PatientID"],

            "StudyInstanceUID": getattr(
                ds,
                "StudyInstanceUID",
                None
            ),

            "SeriesInstanceUID": getattr(
                ds,
                "SeriesInstanceUID",
                None
            ),

            "SOPInstanceUID": getattr(
                ds,
                "SOPInstanceUID",
                None
            ),

            "Modality": getattr(
                ds,
                "Modality",
                None
            ),

            "SOPClassUID": getattr(
                ds,
                "SOPClassUID",
                None
            ),

            "InstanceNumber": getattr(
                ds,
                "InstanceNumber",
                None
            ),

            "ImageLaterality": getattr(
                ds,
                "ImageLaterality",
                None
            ),

            "ViewPosition": getattr(
                ds,
                "ViewPosition",
                None
            ),

            "Rows": getattr(
                ds,
                "Rows",
                None
            ),

            "Columns": getattr(
                ds,
                "Columns",
                None
            ),

            "PhotometricInterpretation": getattr(
                ds,
                "PhotometricInterpretation",
                None
            ),

            "BitsAllocated": getattr(
                ds,
                "BitsAllocated",
                None
            ),

            "BitsStored": getattr(
                ds,
                "BitsStored",
                None
            ),

            "HighBit": getattr(
                ds,
                "HighBit",
                None
            ),

            "PixelRepresentation": getattr(
                ds,
                "PixelRepresentation",
                None
            ),

            "SamplesPerPixel": getattr(
                ds,
                "SamplesPerPixel",
                None
            ),

            "ImageType": str(
                getattr(ds, "ImageType", None)
            ),

            "filename": dcm_path.name,

            "dicom_path": str(dcm_path)
        })

    except Exception as e:

        print(f"\nERROR reading:")
        print(dcm_path)
        print(e)

# ---------------------------------------------------------
# Create dataframe
# ---------------------------------------------------------

dicom_headers = pd.DataFrame(records)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

dicom_headers.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Saved:")
print(OUTPUT_FILE)

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print()
print("Summary")
print("-------")

print(
    "Patients:",
    dicom_headers["PatientID"].nunique()
)

print(
    "Studies:",
    dicom_headers["StudyInstanceUID"].nunique()
)

print(
    "Series:",
    dicom_headers["SeriesInstanceUID"].nunique()
)

print(
    "SOP instances:",
    dicom_headers["SOPInstanceUID"].nunique()
)

print()
print("Modality:")
print(dicom_headers["Modality"].value_counts())

print()
print("ViewPosition:")
print(dicom_headers["ViewPosition"].value_counts(dropna=False))

print()
print("ImageLaterality:")
print(dicom_headers["ImageLaterality"].value_counts(dropna=False))

print()
print("Image dimensions:")
print(
    dicom_headers[
        ["Rows", "Columns"]
    ].value_counts()
)