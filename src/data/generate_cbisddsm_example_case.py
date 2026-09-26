"""
generate_cbisddsm_example_case.py

Generate a report-ready CBIS-DDSM example case containing:
1. full mammogram
2. cropped abnormality image
3. ROI mask
4. ROI overlay on the full mammogram

It also saves:
- a compact metadata CSV
- a LaTeX metadata table
- a four-panel PNG figure

Usage:
    python src/data/generate_cbisddsm_example_case.py
    python src/data/generate_cbisddsm_example_case.py --patient-id P_00674
    python src/data/generate_cbisddsm_example_case.py --patient-id P_00674 --abnormality-id 1
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path.cwd()
DEFAULT_MANIFEST = Path(
    "data/processed/cbis_ddsm/manifests/cbis_ddsm_processed_manifest.csv"
)
OUTPUT_DIR = Path("results/figures/cbisddsm_example_case")


def resolve_path(value):
    if pd.isna(value):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def find_column(df, candidates, required=True):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    if required:
        raise ValueError(
            "Could not find any of these columns:\n"
            + "\n".join(f"  - {c}" for c in candidates)
        )
    return None


def load_dicom_image(path):
    ds = pydicom.dcmread(path)
    image = ds.pixel_array.astype(np.float32)

    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        image = image.max() - image

    low = np.percentile(image, 1)
    high = np.percentile(image, 99)

    if high <= low:
        low = float(image.min())
        high = float(image.max())

    image = np.clip(image, low, high)

    if high > low:
        image = (image - low) / (high - low)
    else:
        image = np.zeros_like(image)

    return image


def load_mask(path):
    if path.suffix.lower() == ".dcm":
        ds = pydicom.dcmread(path)
        array = ds.pixel_array
    else:
        array = np.asarray(Image.open(path).convert("L"))
    return array > 0


def resize_mask(mask, target_shape):
    target_height, target_width = target_shape
    pil_mask = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = pil_mask.resize(
        (target_width, target_height),
        resample=Image.Resampling.NEAREST,
    )
    return np.asarray(resized) > 0


def overlay_roi(image, mask, line_width=6):
    """
    Produce an RGB image with a bold red ROI boundary.

    Parameters
    ----------
    image : numpy.ndarray
        Normalised grayscale mammogram.
    mask : numpy.ndarray
        Binary ROI mask.
    line_width : int
        Approximate thickness of the ROI outline in pixels.
    """

    if mask.shape != image.shape:
        mask = resize_mask(mask, image.shape)

    image_uint8 = np.clip(
        image * 255.0,
        0,
        255
    ).astype(np.uint8)

    rgb = np.stack(
        [image_uint8] * 3,
        axis=-1
    )

    # --------------------------------------------------------
    # Find the original ROI boundary
    # --------------------------------------------------------

    padded = np.pad(
        mask.astype(np.uint8),
        1,
        mode="constant"
    )

    center = padded[1:-1, 1:-1].astype(bool)

    interior = (
        padded[:-2, 1:-1].astype(bool)
        & padded[2:, 1:-1].astype(bool)
        & padded[1:-1, :-2].astype(bool)
        & padded[1:-1, 2:].astype(bool)
    )

    boundary = center & ~interior

    # --------------------------------------------------------
    # Make boundary thicker
    # --------------------------------------------------------

    thick_boundary = boundary.copy()

    for _ in range(line_width):
        padded_boundary = np.pad(
            thick_boundary.astype(np.uint8),
            1,
            mode="constant"
        )

        thick_boundary = (
            padded_boundary[1:-1, 1:-1]
            | padded_boundary[:-2, 1:-1]
            | padded_boundary[2:, 1:-1]
            | padded_boundary[1:-1, :-2]
            | padded_boundary[1:-1, 2:]
            | padded_boundary[:-2, :-2]
            | padded_boundary[:-2, 2:]
            | padded_boundary[2:, :-2]
            | padded_boundary[2:, 2:]
        ).astype(bool)

    # --------------------------------------------------------
    # Draw boundary in bright red
    # --------------------------------------------------------

    rgb[thick_boundary, 0] = 255
    rgb[thick_boundary, 1] = 0
    rgb[thick_boundary, 2] = 0

    return rgb


def latex_escape(value):
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--patient-id", type=str, default=None)
    parser.add_argument("--abnormality-id", type=int, default=None)
    args = parser.parse_args()

    manifest_path = resolve_path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found:\n{manifest_path}")

    df = pd.read_csv(manifest_path)

    patient_col = find_column(df, ["patient_id", "patient id"])
    abnormality_col = find_column(
        df, ["abnormality id", "abnormality_id"], required=False
    )
    abnormality_type_col = find_column(
        df, ["abnormality type", "abnormality_type"]
    )
    pathology_col = find_column(df, ["pathology"])
    laterality_col = find_column(
        df, ["laterality", "left or right breast"]
    )
    view_col = find_column(df, ["image view", "image_view", "view"])
    dataset_col = find_column(df, ["dataset"], required=False)

    full_path_col = find_column(
        df,
        [
            "full_mammogram_path",
            "full_jpeg_path",
            "image file path",
        ]
    )

    crop_path_col = find_column(
        df,
        [
            "cropped_path",
            "cropped_jpeg_path",
            "cropped image file path",
        ]
    )

    roi_path_col = find_column(
        df,
        [
            "roi_mask_path",
            "roi_png_path",
            "ROI mask file path",
        ]
    )

    usable = df[
        (df["has_full_mammogram"] == True)
        & (df["has_cropped"] == True)
        & (df["has_roi_mask"] == True)
        & (df["full_match_status"] == "resolved")
        & (df["cropped_match_status"] == "resolved")
        & (df["roi_match_status"] == "resolved")
    ].copy()

    if args.patient_id is not None:
        usable = usable[
            usable[patient_col].astype(str) == args.patient_id
        ]

    if args.abnormality_id is not None and abnormality_col is not None:
        usable = usable[
            pd.to_numeric(usable[abnormality_col], errors="coerce")
            == args.abnormality_id
        ]

    if len(usable) == 0:
        raise ValueError("No usable case matched the requested criteria.")

    row = usable.iloc[0]

    full_path = resolve_path(row[full_path_col])
    crop_path = resolve_path(row[crop_path_col])
    roi_path = resolve_path(row[roi_path_col])

    for label, path in [
        ("Full mammogram", full_path),
        ("Cropped abnormality", crop_path),
        ("ROI mask", roi_path),
    ]:
        if path is None or not path.exists():
            raise FileNotFoundError(f"{label} not found:\n{path}")

    full_image = load_dicom_image(full_path)
    crop_image = load_dicom_image(crop_path)
    roi_mask = load_mask(roi_path)
    overlay = overlay_roi(full_image, roi_mask)

    metadata = [
        ("Patient ID", row[patient_col]),
        (
            "Dataset group",
            row[dataset_col] if dataset_col is not None else "Not available",
        ),
        ("Abnormality type", row[abnormality_type_col]),
        (
            "Abnormality ID",
            row[abnormality_col]
            if abnormality_col is not None
            else "Not available",
        ),
        ("Pathology", row[pathology_col]),
        ("Laterality", row[laterality_col]),
        ("Image view", row[view_col]),
        ("Full mammogram", full_path.name),
        ("Cropped abnormality", crop_path.name),
        ("ROI mask", roi_path.name),
    ]

    metadata_df = pd.DataFrame(metadata, columns=["Field", "Example value"])

    output_dir = resolve_path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv = output_dir / "cbisddsm_example_metadata.csv"
    metadata_df.to_csv(metadata_csv, index=False)

    figure_path = output_dir / "cbisddsm_example_case.png"

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))

    axes[0].imshow(full_image, cmap="gray")
    axes[0].set_title("Full mammogram")

    axes[1].imshow(crop_image, cmap="gray")
    axes[1].set_title("Cropped abnormality")

    axes[2].imshow(roi_mask, cmap="gray")
    axes[2].set_title("ROI mask")

    axes[3].imshow(overlay)
    axes[3].set_title("Lesion ROI overlay")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{row[patient_col]} | "
        f"{str(row[laterality_col]).title()} "
        f"{str(row[view_col]).upper()} | "
        f"{str(row[pathology_col]).title()} "
        f"{str(row[abnormality_type_col]).lower()}"
    )

    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    latex_rows = []
    for field, value in metadata:
        latex_rows.append(
            f"{latex_escape(field)} & {latex_escape(value)} \\\\"
        )

    latex_table = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{Example CBIS-DDSM abnormality metadata and linked image resources.}\n"
        "\\label{tab:cbis_example_metadata}\n"
        "\\small\n"
        "\\begin{tabularx}{\\textwidth}{@{}lX@{}}\n"
        "\\toprule\n"
        "\\textbf{Field} & \\textbf{Example value} \\\\\n"
        "\\midrule\n"
        + "\n".join(latex_rows)
        + "\n\\bottomrule\n"
        "\\end{tabularx}\n"
        "\\end{table}\n"
    )

    latex_path = output_dir / "cbisddsm_example_metadata_table.tex"
    latex_path.write_text(latex_table, encoding="utf-8")

    print("=" * 72)
    print("CBIS-DDSM EXAMPLE CASE")
    print("=" * 72)
    print()
    print(metadata_df.to_string(index=False))
    print()
    print("Saved figure:")
    print(figure_path)
    print()
    print("Saved metadata CSV:")
    print(metadata_csv)
    print()
    print("Saved LaTeX table:")
    print(latex_path)
    print()
    print("Suggested LaTeX figure:")
    print(
        r"""
\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]
    {results/figures/cbisddsm_example_case/cbisddsm_example_case.png}
    \caption{Example CBIS-DDSM case showing the linked full mammogram,
    cropped abnormality image, ROI mask, and lesion location on the
    full mammogram.}
    \label{fig:cbis_example_case}
\end{figure}
"""
    )


if __name__ == "__main__":
    main()