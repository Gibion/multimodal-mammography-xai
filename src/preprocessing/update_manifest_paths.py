from pathlib import Path
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path.cwd()

CBIS_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cbis_ddsm"
)

FULL_DIR = CBIS_ROOT / "full_mammograms"
CROP_DIR = CBIS_ROOT / "cropped"
ROI_DIR = CBIS_ROOT / "roi_masks"

MANIFEST_DIR = CBIS_ROOT / "manifests"


MANIFEST_FILES = [
    "cbis_ddsm_processed_manifest.csv",
    "cbis_ddsm_qa_manifest.csv",
    "cbis_ddsm_full_image_manifest.csv",
    "cbis_ddsm_full_image_manifest_split.csv",
]


# ============================================================
# Path mapping
# ============================================================

PATH_COLUMNS = {
    "full_jpeg_path": FULL_DIR,
    "cropped_jpeg_path": CROP_DIR,
    "roi_png_path": ROI_DIR,
}


def update_path(value, target_directory):
    """
    Replace an old generated-image path with its current
    location, preserving the filename.
    """

    if pd.isna(value):
        return value

    old_path = Path(str(value))

    new_path = (
        target_directory
        / old_path.name
    )

    return str(
        new_path.relative_to(PROJECT_ROOT)
    )


# ============================================================
# Update manifests
# ============================================================

for manifest_name in MANIFEST_FILES:

    manifest_path = (
        MANIFEST_DIR / manifest_name
    )

    if not manifest_path.exists():

        print(
            f"Skipping missing manifest: "
            f"{manifest_path}"
        )

        continue

    print()
    print("=" * 72)
    print(manifest_name)
    print("=" * 72)

    df = pd.read_csv(manifest_path)

    for column, directory in PATH_COLUMNS.items():

        if column not in df.columns:
            continue

        df[column] = df[column].apply(
            lambda x: update_path(
                x,
                directory
            )
        )

        existing = (
            df[column]
            .dropna()
            .apply(
                lambda x: Path(x).exists()
            )
        )

        print(
            f"{column}: "
            f"{existing.sum():,} / "
            f"{len(existing):,} exist"
        )

    df.to_csv(
        manifest_path,
        index=False
    )

    print("Updated:", manifest_path)