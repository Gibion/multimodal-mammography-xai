from pathlib import Path
import pandas as pd

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

METADATA_FILE = "../raw/CBIS-DDSM-All-doiJNLP-zzWs5zfZ/metadata/metadata.csv"

# ---------------------------------------------------------
# Load metadata
# ---------------------------------------------------------

metadata = pd.read_csv(METADATA_FILE)

print(f"Metadata records: {len(metadata):,}")
print()

# ---------------------------------------------------------
# Check S5cmdManifestPath
# ---------------------------------------------------------

exists = []
is_file = []
is_directory = []

for path_string in metadata["S5cmdManifestPath"]:

    path = Path(path_string)

    exists.append(path.exists())
    is_file.append(path.is_file())
    is_directory.append(path.is_dir())

metadata["path_exists"] = exists
metadata["path_is_file"] = is_file
metadata["path_is_directory"] = is_directory

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("Path verification")
print("-----------------")
print(f"Paths that exist:       {metadata['path_exists'].sum():,}")
print(f"Paths that do NOT exist:{(~metadata['path_exists']).sum():,}")
print()
print(f"Paths that are files:   {metadata['path_is_file'].sum():,}")
print(f"Paths that are folders: {metadata['path_is_directory'].sum():,}")

# ---------------------------------------------------------
# Show missing paths
# ---------------------------------------------------------

missing = metadata[~metadata["path_exists"]]

if len(missing) > 0:
    print("\nMISSING PATHS")
    print("-------------")

    print(
        missing[
            [
                "PatientID",
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "S5cmdManifestPath"
            ]
        ].to_string(index=False)
    )

# ---------------------------------------------------------
# Save verification results
# ---------------------------------------------------------

metadata.to_csv(
    "metadata_path_verification.csv",
    index=False
)

print("\nSaved:")
print("metadata_path_verification.csv")