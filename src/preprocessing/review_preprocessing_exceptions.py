from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


MANIFEST = Path(
    "data/processed/cbis_ddsm/manifests/"
    "cbis_ddsm_full_image_manifest_preprocessed.csv"
)

QA_MANIFEST = Path(
    "data/processed/cbis_ddsm/manifests/"
    "cbis_ddsm_full_image_preprocessing_qa.csv"
)

OUTPUT_DIR = Path(
    "results/figures/preprocessing_qa"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def resolve_path(value):
    path = Path(str(value))

    if not path.is_absolute():
        path = Path.cwd() / path

    return path


def load_image(path):
    with Image.open(path) as img:
        return np.asarray(
            img.convert("L")
        )


# ============================================================
# Load data
# ============================================================

manifest = pd.read_csv(MANIFEST)
qa = pd.read_csv(QA_MANIFEST)

# Join QA results back to the main manifest
qa = qa.rename(
    columns={
        "path": "qa_preprocessed_path"
    }
)

# Normalise paths before matching
manifest["merge_path"] = (
    manifest["preprocessed_path"]
    .astype(str)
    .map(lambda x: str(resolve_path(x).resolve()))
)

qa["merge_path"] = (
    qa["qa_preprocessed_path"]
    .astype(str)
    .map(lambda x: str(resolve_path(x).resolve()))
)

df = manifest.merge(
    qa[
        [
            "image_id",
            "nonzero_fraction",
            "orientation_score",
            "std_pixel"
        ]
    ],
    on="image_id",
    how="left",
    validate="one_to_one"
)


# ============================================================
# Helper for creating galleries
# ============================================================

def save_gallery(
    subset,
    output_name,
    title,
    max_images=20
):

    subset = subset.head(
        max_images
    )

    if len(subset) == 0:
        print(
            f"No images for: {title}"
        )
        return

    cols = 4
    rows = int(
        np.ceil(
            len(subset) / cols
        )
    )

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(
            12,
            rows * 4
        )
    )

    axes = np.atleast_1d(
        axes
    ).flatten()

    for ax in axes:
        ax.axis("off")

    for ax, (_, row) in zip(
        axes,
        subset.iterrows()
    ):

        image_path = resolve_path(
            row["preprocessed_path"]
        )

        image = load_image(
            image_path
        )

        ax.imshow(
            image,
            cmap="gray"
        )

        ax.axis("off")

        ax.set_title(
            f"{row['patient_id']}\n"
            f"{row['binary_pathology']} | "
            f"{row['image_view']}\n"
            f"foreground={row['nonzero_fraction']:.3f}\n"
            f"orientation={row['orientation_score']:.3f}",
            fontsize=8
        )

    fig.suptitle(
        title,
        fontsize=14
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / output_name
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "Saved:",
        output_path.resolve()
    )


# ============================================================
# 1. Orientation exceptions
# ============================================================

orientation_exceptions = (
    df[
        df["orientation_score"] < 0
    ]
    .sort_values(
        "orientation_score"
    )
)

print(
    "\nOrientation exceptions:",
    len(orientation_exceptions)
)

print(
    orientation_exceptions[
        [
            "patient_id",
            "image_view",
            "binary_pathology",
            "orientation_score",
            "nonzero_fraction",
            "chest_wall_original",
            "orientation_flipped",
        ]
    ].to_string(index=False)
)

save_gallery(
    orientation_exceptions,
    "orientation_exceptions.png",
    "Post-normalisation orientation exceptions",
    max_images=20
)


# ============================================================
# 2. Lowest foreground fractions
# ============================================================

lowest_foreground = (
    df.sort_values(
        "nonzero_fraction",
        ascending=True
    )
)

save_gallery(
    lowest_foreground,
    "lowest_foreground_fraction.png",
    "Images with the lowest foreground fraction",
    max_images=20
)


# ============================================================
# 3. Highest foreground fractions
# ============================================================

highest_foreground = (
    df.sort_values(
        "nonzero_fraction",
        ascending=False
    )
)

save_gallery(
    highest_foreground,
    "highest_foreground_fraction.png",
    "Images with the highest foreground fraction",
    max_images=20
)

print("\nRows after merge:", len(df))

print(
    "Unique preprocessed images:",
    df["preprocessed_path"].nunique()
)

print(
    "Missing QA matches:",
    df["orientation_score"].isna().sum()
)

orientation_exceptions = (
    df[df["orientation_score"] < 0]
    .sort_values("orientation_score")
)

print(
    "\nTrue image-level orientation exceptions:",
    len(orientation_exceptions)
)