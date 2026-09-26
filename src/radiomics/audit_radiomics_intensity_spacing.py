from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

PROJECT_ROOT = Path.cwd()

MANIFEST_CANDIDATES = [
    Path("data/processed/cbis_ddsm/manifests/cbis_ddsm_processed_manifest.csv"),
    Path("data/processed/cbis_ddsm/cbis_ddsm_processed_manifest.csv"),
]

OUTPUT_CSV = Path(
    "data/processed/cbis_ddsm/manifests/"
    "cbis_ddsm_radiomics_intensity_spacing_audit.csv"
)

SUMMARY_TXT = Path(
    "results/radiomics/radiomics_intensity_spacing_summary.txt"
)

FULL_IMAGE_COLUMN = "full_mammogram_path"


def find_manifest():
    for path in MANIFEST_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not locate cbis_ddsm_processed_manifest.csv."
    )


def resolve_path(value):
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def get_pair(ds, name):
    value = getattr(ds, name, None)
    if value is None:
        return np.nan, np.nan
    try:
        return float(value[0]), float(value[1])
    except Exception:
        return np.nan, np.nan


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


manifest_path = find_manifest()
df = pd.read_csv(manifest_path)

if FULL_IMAGE_COLUMN not in df.columns:
    raise ValueError(
        f"Missing required column: {FULL_IMAGE_COLUMN}"
    )

keep_cols = [FULL_IMAGE_COLUMN]
for c in ["patient_id", "dataset"]:
    if c in df.columns:
        keep_cols.append(c)

full_df = (
    df[keep_cols]
    .dropna(subset=[FULL_IMAGE_COLUMN])
    .drop_duplicates(subset=[FULL_IMAGE_COLUMN])
    .reset_index(drop=True)
)

print("=" * 72)
print("CBIS-DDSM RADIOMICS INTENSITY / PIXEL-SPACING AUDIT")
print("=" * 72)
print(f"\nManifest: {manifest_path.resolve()}")
print(f"Unique full mammograms: {len(full_df):,}")

records = []

for _, row in tqdm(
    full_df.iterrows(),
    total=len(full_df),
    desc="Auditing mammograms"
):
    path = resolve_path(row[FULL_IMAGE_COLUMN])

    record = {
        "patient_id": row.get("patient_id", ""),
        "dataset": row.get("dataset", ""),
        "full_mammogram_path": str(path),
        "exists": path.exists(),
    }

    if not path.exists():
        record["audit_status"] = "missing_file"
        records.append(record)
        continue

    try:
        ds = pydicom.dcmread(str(path))
        pixels = np.asarray(ds.pixel_array)
        pixels = np.squeeze(pixels)

        finite = pixels[np.isfinite(pixels)].astype(np.float64)

        if finite.size == 0:
            raise ValueError("No finite pixel values.")

        pixel_spacing = get_pair(ds, "PixelSpacing")
        imager_spacing = get_pair(ds, "ImagerPixelSpacing")
        nominal_spacing = get_pair(ds, "NominalScannedPixelSpacing")

        record.update({
            "audit_status": "success",
            "Rows": int(getattr(ds, "Rows", pixels.shape[0])),
            "Columns": int(getattr(ds, "Columns", pixels.shape[1])),
            "BitsAllocated": getattr(ds, "BitsAllocated", np.nan),
            "BitsStored": getattr(ds, "BitsStored", np.nan),
            "PhotometricInterpretation": str(
                getattr(ds, "PhotometricInterpretation", "")
            ),
            "pixel_min": float(finite.min()),
            "pixel_max": float(finite.max()),
            "pixel_mean": float(finite.mean()),
            "pixel_std": float(finite.std()),
            "pixel_p01": float(np.percentile(finite, 1)),
            "pixel_p05": float(np.percentile(finite, 5)),
            "pixel_p50": float(np.percentile(finite, 50)),
            "pixel_p95": float(np.percentile(finite, 95)),
            "pixel_p99": float(np.percentile(finite, 99)),
            "n_unique_pixel_values": int(np.unique(finite).size),
            "RescaleSlope": safe_float(
                getattr(ds, "RescaleSlope", None)
            ),
            "RescaleIntercept": safe_float(
                getattr(ds, "RescaleIntercept", None)
            ),
            "PixelSpacing_row": pixel_spacing[0],
            "PixelSpacing_col": pixel_spacing[1],
            "ImagerPixelSpacing_row": imager_spacing[0],
            "ImagerPixelSpacing_col": imager_spacing[1],
            "NominalScannedPixelSpacing_row": nominal_spacing[0],
            "NominalScannedPixelSpacing_col": nominal_spacing[1],
        })

    except Exception as exc:
        record["audit_status"] = f"error: {exc}"

    records.append(record)

audit = pd.DataFrame(records)

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
audit.to_csv(OUTPUT_CSV, index=False)

success = audit[audit["audit_status"] == "success"].copy()

lines = []

def show(text=""):
    print(text)
    lines.append(str(text))


show("\n" + "=" * 72)
show("AUDIT SUMMARY")
show("=" * 72)

show(
    f"\nSuccessful audits: {len(success):,} / {len(audit):,}"
)

for column in [
    "BitsAllocated",
    "BitsStored",
    "PhotometricInterpretation",
]:
    show(f"\n{column}:")
    show(success[column].value_counts(dropna=False).to_string())

for column in [
    "pixel_min",
    "pixel_max",
    "pixel_mean",
    "pixel_std",
    "pixel_p99",
    "n_unique_pixel_values",
]:
    show(f"\n{column}:")
    show(success[column].describe().to_string())

spacing_sets = [
    ("PixelSpacing", "PixelSpacing_row", "PixelSpacing_col"),
    ("ImagerPixelSpacing", "ImagerPixelSpacing_row", "ImagerPixelSpacing_col"),
    (
        "NominalScannedPixelSpacing",
        "NominalScannedPixelSpacing_row",
        "NominalScannedPixelSpacing_col",
    ),
]

for label, row_col, col_col in spacing_sets:
    available = success[
        success[row_col].notna()
        & success[col_col].notna()
    ]

    show(
        f"\n{label} available: "
        f"{len(available):,} / {len(success):,}"
    )

    if len(available):
        show("Row spacing:")
        show(available[row_col].describe().to_string())
        show("Column spacing:")
        show(available[col_col].describe().to_string())

        ratio = available[row_col] / available[col_col]
        show("Row/column spacing ratio:")
        show(ratio.describe().to_string())

show("\nIntensity statistics by BitsAllocated:")

for bits, group in success.groupby("BitsAllocated"):
    show(f"\nBitsAllocated = {bits}")
    show(
        group[
            [
                "pixel_min",
                "pixel_max",
                "pixel_p01",
                "pixel_p50",
                "pixel_p99",
                "pixel_mean",
                "pixel_std",
                "n_unique_pixel_values",
            ]
        ].describe().to_string()
    )

show("\n" + "=" * 72)
show("=" * 72)

SUMMARY_TXT.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")

show("\nSaved audit CSV:")
show(OUTPUT_CSV.resolve())

show("\nSaved summary:")
show(SUMMARY_TXT.resolve())
