from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


MANIFEST = Path(
    "data/processed/cbis_ddsm/manifests/"
    "cbis_ddsm_full_image_manifest_preprocessed_v3.csv"
)

OUTPUT_DIR = Path(
    "results/figures/preprocessing_qa_v3"
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


def load_gray(path):
    with Image.open(path) as img:
        return np.asarray(img.convert("L"))


def save_gallery(
    subset,
    filename,
    title,
    max_images=20
):
    subset = subset.head(max_images)

    if len(subset) == 0:
        return

    cols = 4
    rows = math.ceil(len(subset) / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(12, rows * 4)
    )

    axes = np.atleast_1d(axes).flatten()

    for ax in axes:
        ax.axis("off")

    for ax, (_, row) in zip(
        axes,
        subset.iterrows()
    ):
        image = load_gray(
            resolve_path(
                row["preprocessed_path"]
            )
        )

        ax.imshow(
            image,
            cmap="gray"
        )

        ax.axis("off")

        ax.set_title(
            f"{row['patient_id']} | "
            f"{row['laterality']} | "
            f"{row['image_view']}\n"
            f"removed={float(row['total_removed_fraction']):.3f}\n"
            f"L/R/T/B="
            f"{float(row['crop_left_fraction']):.2f}/"
            f"{float(row['crop_right_fraction']):.2f}/"
            f"{float(row['crop_top_fraction']):.2f}/"
            f"{float(row['crop_bottom_fraction']):.2f}",
            fontsize=8
        )

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    output = OUTPUT_DIR / filename

    fig.savefig(
        output,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", output.resolve())


df = pd.read_csv(MANIFEST)

success = df[
    df["preprocessing_status"] == "success"
].copy()

for column in [
    "total_removed_fraction",
    "crop_left_fraction",
    "crop_right_fraction",
    "crop_top_fraction",
    "crop_bottom_fraction",
]:
    success[column] = pd.to_numeric(
        success[column],
        errors="coerce"
    )

print("=" * 72)
print("V3 CROP QA")
print("=" * 72)

print("\nSuccessful rows:")
print(len(success))

print("\nMost aggressively cropped images:")
print(
    success[
        [
            "patient_id",
            "laterality",
            "image_view",
            "total_removed_fraction",
            "crop_left_fraction",
            "crop_right_fraction",
            "crop_top_fraction",
            "crop_bottom_fraction",
        ]
    ]
    .sort_values(
        "total_removed_fraction",
        ascending=False
    )
    .head(30)
    .to_string(index=False)
)

save_gallery(
    success.sort_values(
        "total_removed_fraction",
        ascending=False
    ),
    "most_aggressively_cropped.png",
    "V3: Most Aggressively Cropped Mammograms"
)

save_gallery(
    success.sort_values(
        "total_removed_fraction",
        ascending=True
    ),
    "least_cropped.png",
    "V3: Least Cropped Mammograms"
)

chest_wall_crop = success[
    (
        (success["laterality"] == "LEFT")
        & (success["crop_left_fraction"] > 0)
    )
    |
    (
        (success["laterality"] == "RIGHT")
        & (success["crop_right_fraction"] > 0)
    )
]

print("\nImages where chest-wall edge was cropped:")
print(len(chest_wall_crop))
