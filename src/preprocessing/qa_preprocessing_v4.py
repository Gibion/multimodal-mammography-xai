from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


MANIFEST = Path(
    "data/processed/cbis_ddsm/manifests/"
    "cbis_ddsm_full_image_manifest_preprocessed_v4.csv"
)

OUTPUT_DIR = Path(
    "results/figures/preprocessing_qa_v4"
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
        print(f"No images to plot for {title}")
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

    output_path = OUTPUT_DIR / filename

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", output_path.resolve())


df = pd.read_csv(MANIFEST)

success = df[
    df["preprocessing_status"] == "success"
].copy()

numeric_columns = [
    "total_removed_fraction",
    "crop_left_fraction",
    "crop_right_fraction",
    "crop_top_fraction",
    "crop_bottom_fraction",
    "foreground_fraction_before_crop",
    "foreground_fraction_after_crop",
]

for column in numeric_columns:
    success[column] = pd.to_numeric(
        success[column],
        errors="coerce"
    )

print("=" * 72)
print("V4 PREPROCESSING QA")
print("=" * 72)

print("\nSuccessful rows:")
print(len(success))

print("\nUnique preprocessed paths:")
print(
    success["preprocessed_path"].nunique()
)

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

vertical_crop = success[
    (success["crop_top_fraction"] > 0)
    |
    (success["crop_bottom_fraction"] > 0)
]

chest_wall_crop = success[
    (
        (success["laterality"] == "LEFT")
        &
        (success["crop_left_fraction"] > 0)
    )
    |
    (
        (success["laterality"] == "RIGHT")
        &
        (success["crop_right_fraction"] > 0)
    )
]

too_much_lateral_crop = success[
    (success["crop_left_fraction"] > 0.2501)
    |
    (success["crop_right_fraction"] > 0.2501)
]

print("\nImages with vertical cropping:")
print(len(vertical_crop))

print("\nImages where chest-wall edge was cropped:")
print(len(chest_wall_crop))

print("\nImages exceeding 25% lateral crop:")
print(len(too_much_lateral_crop))


records = []

for _, row in success.iterrows():

    path = resolve_path(
        row["preprocessed_path"]
    )

    record = {
        "image_id": row["image_id"],
        "patient_id": row["patient_id"],
        "exists": path.exists(),
    }

    if path.exists():
        image = load_gray(path)

        record.update({
            "width": image.shape[1],
            "height": image.shape[0],
            "min_pixel": int(image.min()),
            "max_pixel": int(image.max()),
            "mean_pixel": float(image.mean()),
            "std_pixel": float(image.std()),
            "nonzero_fraction": float(
                np.count_nonzero(image)
                / image.size
            ),
        })

    records.append(record)

qa_df = pd.DataFrame(records)

print("\nFiles found:")
print(
    qa_df["exists"].value_counts(dropna=False)
)

valid = qa_df[
    qa_df["exists"] == True
]

print("\nImage sizes:")
print(
    valid[
        ["width", "height"]
    ].value_counts()
)

print("\nBlank / near-blank images (std < 1):")
print(
    (valid["std_pixel"] < 1).sum()
)


save_gallery(
    success.sort_values(
        "total_removed_fraction",
        ascending=False
    ),
    "most_aggressively_cropped.png",
    "V4: Most Aggressively Cropped Mammograms"
)

save_gallery(
    success.sort_values(
        "total_removed_fraction",
        ascending=True
    ),
    "least_cropped.png",
    "V4: Least Cropped Mammograms"
)

save_gallery(
    success.sort_values(
        "foreground_fraction_after_crop",
        ascending=True
    ),
    "lowest_foreground_fraction.png",
    "V4: Lowest Foreground Fraction"
)

save_gallery(
    success.sort_values(
        "foreground_fraction_after_crop",
        ascending=False
    ),
    "highest_foreground_fraction.png",
    "V4: Highest Foreground Fraction"
)


QA_MANIFEST = Path(
    "data/processed/cbis_ddsm/manifests/"
    "cbis_ddsm_full_image_preprocessing_qa_v4.csv"
)

qa_df.to_csv(
    QA_MANIFEST,
    index=False
)

print("\nSaved QA manifest:")
print(QA_MANIFEST.resolve())

print("\nQA figures:")
print(OUTPUT_DIR.resolve())
