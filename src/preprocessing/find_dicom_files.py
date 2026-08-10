from pathlib import Path
import pandas as pd

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

METADATA_FILE = "../raw/CBIS-DDSM-All-doiJNLP-zzWs5zfZ/metadata/metadata.csv"
OUTPUT_FILE = "dicom_manifest.csv"

# ---------------------------------------------------------
# Load metadata
# ---------------------------------------------------------

metadata = pd.read_csv(METADATA_FILE)

print(f"Metadata records: {len(metadata):,}")
print()

# ---------------------------------------------------------
# Find DICOM files inside each S5cmdManifestPath
# ---------------------------------------------------------

records = []

for index, row in metadata.iterrows():

    directory = Path(row["S5cmdManifestPath"])

    # Find DICOM files directly inside the directory
    dcm_files = list(directory.glob("*.dcm"))

    # If none are found directly, search recursively
    if len(dcm_files) == 0:
        dcm_files = list(directory.rglob("*.dcm"))

    for dcm_file in dcm_files:

        records.append({
            "metadata_index": index,
            "PatientID": row["PatientID"],
            "StudyInstanceUID": row["StudyInstanceUID"],
            "SeriesInstanceUID": row["SeriesInstanceUID"],
            "Collection": row["Collection"],
            "FileSize": row["FileSize"],
            "S5cmdManifestPath": row["S5cmdManifestPath"],
            "dicom_path": str(dcm_file),
            "dicom_filename": dcm_file.name
        })

# ---------------------------------------------------------
# Create dataframe
# ---------------------------------------------------------

dicom_manifest = pd.DataFrame(records)

# ---------------------------------------------------------
# Display summary
# ---------------------------------------------------------

print("DICOM discovery")
print("----------------")

print(f"Metadata records:       {len(metadata):,}")
print(f"DICOM files discovered: {len(dicom_manifest):,}")

# ---------------------------------------------------------
# Check how many metadata records have DICOM files
# ---------------------------------------------------------

matched_metadata = dicom_manifest["metadata_index"].nunique()

print(f"Metadata records with DICOM: {matched_metadata:,}")
print(
    f"Metadata records WITHOUT DICOM: "
    f"{len(metadata) - matched_metadata:,}"
)

# ---------------------------------------------------------
# Check number of DICOMs per metadata record
# ---------------------------------------------------------

counts = (
    dicom_manifest
    .groupby("metadata_index")
    .size()
)

print("\nDICOMs per metadata record:")
print(counts.value_counts().sort_index())

# ---------------------------------------------------------
# Save manifest
# ---------------------------------------------------------

dicom_manifest.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nSaved:")
print(OUTPUT_FILE)