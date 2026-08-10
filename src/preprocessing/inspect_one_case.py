from pathlib import Path
import pandas as pd
import pydicom

MANIFEST_FILE = "dicom_manifest.csv"

manifest = pd.read_csv(MANIFEST_FILE)

# ---------------------------------------------------------
# Select metadata record 0
# ---------------------------------------------------------

case = manifest[manifest["metadata_index"] == 0]

print("DICOM files for metadata_index = 0")
print("=" * 70)

for _, row in case.iterrows():

    path = Path(row["dicom_path"])

    print("\nFILE:")
    print(path)

    ds = pydicom.dcmread(path, stop_before_pixels=True)

    print("\nDICOM metadata")
    print("-" * 70)

    tags = [
        ("PatientID", "PatientID"),
        ("StudyInstanceUID", "StudyInstanceUID"),
        ("SeriesInstanceUID", "SeriesInstanceUID"),
        ("SOPInstanceUID", "SOPInstanceUID"),
        ("SOPClassUID", "SOPClassUID"),
        ("Modality", "Modality"),
        ("SeriesDescription", "SeriesDescription"),
        ("StudyDescription", "StudyDescription"),
        ("InstanceNumber", "InstanceNumber"),
        ("ImageType", "ImageType"),
        ("Rows", "Rows"),
        ("Columns", "Columns"),
        ("BitsAllocated", "BitsAllocated"),
        ("BitsStored", "BitsStored"),
        ("HighBit", "HighBit"),
        ("PixelRepresentation", "PixelRepresentation"),
        ("SamplesPerPixel", "SamplesPerPixel"),
        ("PhotometricInterpretation", "PhotometricInterpretation"),
        ("TransferSyntaxUID", "TransferSyntaxUID"),
        ("Manufacturer", "Manufacturer"),
        ("ManufacturerModelName", "ManufacturerModelName"),
        ("BodyPartExamined", "BodyPartExamined"),
        ("Laterality", "Laterality"),
        ("ImageLaterality", "ImageLaterality"),
        ("ViewPosition", "ViewPosition"),
        ("PresentationIntentType", "PresentationIntentType"),
    ]

    for label, attribute in tags:
        value = getattr(ds, attribute, "<not present>")
        print(f"{label:30}: {value}")

    print("\nAll available DICOM tags")
    print("-" * 70)

    for element in ds:
        print(element)