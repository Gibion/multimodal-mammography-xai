from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = Path(
    "results/resnet50v2_finetuned/gradcam_roi_alignment"
)

METRICS_FILE = (
    RESULTS_DIR
    / "gradcam_roi_metrics_per_image.csv"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "final_analysis"
)

FIGURES_DIR = (
    OUTPUT_DIR
    / "figures"
)

TABLES_DIR = (
    OUTPUT_DIR
    / "tables"
)

for directory in [
    OUTPUT_DIR,
    FIGURES_DIR,
    TABLES_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# Load per-image XAI metrics
# ============================================================

if not METRICS_FILE.exists():
    raise FileNotFoundError(
        "Per-image Grad-CAM/ROI metrics file was not found:\n"
        f"{METRICS_FILE.resolve()}"
    )

df = pd.read_csv(
    METRICS_FILE
)

required_columns = [
    "image_id",
    "patient_id",
    "outcome",
    "lesion_group",
    "roi_size_group",
    "pointing_game_hit",
    "energy_inside_roi",
    "inside_outside_ratio",
    "iou_top_10_percent",
    "iou_top_20_percent",
    "iou_top_30_percent",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Metrics file is missing required columns: "
        + ", ".join(
            missing_columns
        )
    )


print("=" * 72)
print("FINAL GRAD-CAM / ROI SUBGROUP ANALYSIS")
print("=" * 72)

print(
    f"\nPer-image rows: "
    f"{len(df):,}"
)


# ============================================================
# Helper functions
# ============================================================

def subgroup_summary(
    frame,
    group_column,
    group_order=None,
):
    grouped = (
        frame
        .groupby(
            group_column,
            observed=True
        )
    )

    rows = []

    for group_name, group in grouped:

        rows.append({
            "group_type": group_column,
            "group": str(
                group_name
            ),
            "n": int(
                len(
                    group
                )
            ),
            "pointing_game_accuracy": float(
                group[
                    "pointing_game_hit"
                ].mean()
            ),
            "mean_energy_inside_roi": float(
                group[
                    "energy_inside_roi"
                ].mean()
            ),
            "median_energy_inside_roi": float(
                group[
                    "energy_inside_roi"
                ].median()
            ),
            "mean_activation_ratio": float(
                group[
                    "inside_outside_ratio"
                ].mean()
            ),
            "median_activation_ratio": float(
                group[
                    "inside_outside_ratio"
                ].median()
            ),
            "mean_iou_top_10": float(
                group[
                    "iou_top_10_percent"
                ].mean()
            ),
            "median_iou_top_10": float(
                group[
                    "iou_top_10_percent"
                ].median()
            ),
            "mean_iou_top_20": float(
                group[
                    "iou_top_20_percent"
                ].mean()
            ),
            "median_iou_top_20": float(
                group[
                    "iou_top_20_percent"
                ].median()
            ),
            "mean_iou_top_30": float(
                group[
                    "iou_top_30_percent"
                ].mean()
            ),
            "median_iou_top_30": float(
                group[
                    "iou_top_30_percent"
                ].median()
            ),
        })

    result = pd.DataFrame(
        rows
    )

    if group_order is not None:
        order_map = {
            value: index
            for index, value
            in enumerate(
                group_order
            )
        }

        result[
            "_sort"
        ] = (
            result[
                "group"
            ]
            .map(
                order_map
            )
            .fillna(
                999
            )
        )

        result = (
            result
            .sort_values(
                "_sort"
            )
            .drop(
                columns="_sort"
            )
            .reset_index(
                drop=True
            )
        )

    return result


def overall_summary(
    frame
):
    return pd.DataFrame([
        {
            "group_type": "overall",
            "group": "All test mammograms",
            "n": int(
                len(
                    frame
                )
            ),
            "pointing_game_accuracy": float(
                frame[
                    "pointing_game_hit"
                ].mean()
            ),
            "mean_energy_inside_roi": float(
                frame[
                    "energy_inside_roi"
                ].mean()
            ),
            "median_energy_inside_roi": float(
                frame[
                    "energy_inside_roi"
                ].median()
            ),
            "mean_activation_ratio": float(
                frame[
                    "inside_outside_ratio"
                ].mean()
            ),
            "median_activation_ratio": float(
                frame[
                    "inside_outside_ratio"
                ].median()
            ),
            "mean_iou_top_10": float(
                frame[
                    "iou_top_10_percent"
                ].mean()
            ),
            "median_iou_top_10": float(
                frame[
                    "iou_top_10_percent"
                ].median()
            ),
            "mean_iou_top_20": float(
                frame[
                    "iou_top_20_percent"
                ].mean()
            ),
            "median_iou_top_20": float(
                frame[
                    "iou_top_20_percent"
                ].median()
            ),
            "mean_iou_top_30": float(
                frame[
                    "iou_top_30_percent"
                ].mean()
            ),
            "median_iou_top_30": float(
                frame[
                    "iou_top_30_percent"
                ].median()
            ),
        }
    ])


def save_boxplot(
    frame,
    group_column,
    value_column,
    group_order,
    title,
    ylabel,
    filename,
    percent=False,
    log_scale=False,
):
    data = []

    labels = []

    for group_name in group_order:

        values = (
            frame.loc[
                frame[
                    group_column
                ] == group_name,
                value_column,
            ]
            .dropna()
            .to_numpy()
        )

        if percent:
            values = (
                values
                * 100.0
            )

        if len(
            values
        ) == 0:
            continue

        data.append(
            values
        )

        labels.append(
            str(
                group_name
            )
        )

    fig = plt.figure(
        figsize=(
            max(
                6,
                1.6
                * len(
                    labels
                )
            ),
            5.2,
        )
    )

    plt.boxplot(
        data,
        tick_labels=labels,
        showfliers=False,
        showmeans=True,
    )

    if log_scale:
        plt.yscale(
            "symlog",
            linthresh=0.05,
        )

    plt.xlabel(
        group_column
        .replace(
            "_",
            " "
        )
        .title()
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_pointing_game_bar_chart(
    summary_frame,
    title,
    filename,
):
    labels = summary_frame[
        "group"
    ].tolist()

    values = (
        summary_frame[
            "pointing_game_accuracy"
        ].to_numpy()
        * 100.0
    )

    fig = plt.figure(
        figsize=(
            max(
                6,
                1.6
                * len(
                    labels
                )
            ),
            5.2,
        )
    )

    bars = plt.bar(
        labels,
        values,
    )

    plt.ylabel(
        "Pointing-Game Accuracy (%)"
    )

    plt.title(
        title
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        values
    ):
        plt.text(
            bar.get_x()
            + bar.get_width()
            / 2.0,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Build subgroup summaries
# ============================================================

overall = overall_summary(
    df
)

by_outcome = subgroup_summary(
    df,
    "outcome",
    group_order=[
        "TP",
        "TN",
        "FP",
        "FN",
    ],
)

by_lesion = subgroup_summary(
    df,
    "lesion_group",
    group_order=[
        "mass",
        "calcification",
        "mixed",
    ],
)

by_roi_size = subgroup_summary(
    df,
    "roi_size_group",
    group_order=[
        "small",
        "medium",
        "large",
    ],
)

combined_summary = pd.concat(
    [
        overall,
        by_outcome,
        by_lesion,
        by_roi_size,
    ],
    ignore_index=True,
)

combined_summary.to_csv(
    TABLES_DIR
    / "xai_subgroup_summary_full.csv",
    index=False,
)


# ============================================================
# Clean report table
# ============================================================

clean_table = combined_summary[
    [
        "group_type",
        "group",
        "n",
        "pointing_game_accuracy",
        "median_energy_inside_roi",
        "median_activation_ratio",
        "median_iou_top_10",
    ]
].copy()

clean_table[
    "pointing_game_accuracy_percent"
] = (
    clean_table[
        "pointing_game_accuracy"
    ]
    * 100.0
)

clean_table[
    "median_energy_inside_roi_percent"
] = (
    clean_table[
        "median_energy_inside_roi"
    ]
    * 100.0
)

clean_table[
    "median_iou_top_10_percent_value"
] = (
    clean_table[
        "median_iou_top_10"
    ]
    * 100.0
)

clean_table = clean_table[
    [
        "group_type",
        "group",
        "n",
        "pointing_game_accuracy_percent",
        "median_energy_inside_roi_percent",
        "median_activation_ratio",
        "median_iou_top_10_percent_value",
    ]
]

clean_table.columns = [
    "Group type",
    "Group",
    "N",
    "Pointing-game accuracy (%)",
    "Median energy inside ROI (%)",
    "Median activation ratio",
    "Median IoU, top 10% (%)",
]

clean_table.to_csv(
    TABLES_DIR
    / "xai_subgroup_summary_report.csv",
    index=False,
)


# ============================================================
# LaTeX table
# ============================================================

latex_table = clean_table.copy()

latex_table[
    "Pointing-game accuracy (%)"
] = latex_table[
    "Pointing-game accuracy (%)"
].map(
    lambda value: f"{value:.1f}"
)

latex_table[
    "Median energy inside ROI (%)"
] = latex_table[
    "Median energy inside ROI (%)"
].map(
    lambda value: f"{value:.3f}"
)

latex_table[
    "Median activation ratio"
] = latex_table[
    "Median activation ratio"
].map(
    lambda value: f"{value:.3f}"
)

latex_table[
    "Median IoU, top 10% (%)"
] = latex_table[
    "Median IoU, top 10% (%)"
].map(
    lambda value: f"{value:.3f}"
)

latex_path = (
    TABLES_DIR
    / "xai_subgroup_summary_table.tex"
)

with latex_path.open(
    "w",
    encoding="utf-8",
) as handle:

    handle.write(
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Grad-CAM localisation results overall and by subgroup. "
        "Median values are reported for the highly skewed continuous "
        "localisation measures.}\n"
        "\\label{tab:gradcam-subgroup-results}\n"
        "\\small\n"
        "\\begin{tabular}{llrrrrr}\n"
        "\\hline\n"
        "Group type & Group & $N$ & PG (\\%) & "
        "Median $E_{\\mathrm{ROI}}$ (\\%) & "
        "Median $R_{\\mathrm{activation}}$ & "
        "Median IoU$_{10}$ (\\%) \\\\\n"
        "\\hline\n"
    )

    previous_group_type = None

    for _, row in latex_table.iterrows():

        group_type = str(
            row[
                "Group type"
            ]
        )

        if (
            previous_group_type
            is not None
            and
            group_type
            != previous_group_type
        ):
            handle.write(
                "\\hline\n"
            )

        handle.write(
            f"{group_type.replace('_', ' ').title()} & "
            f"{str(row['Group']).replace('_', ' ').title()} & "
            f"{int(row['N'])} & "
            f"{row['Pointing-game accuracy (%)']} & "
            f"{row['Median energy inside ROI (%)']} & "
            f"{row['Median activation ratio']} & "
            f"{row['Median IoU, top 10% (%)']} \\\\\n"
        )

        previous_group_type = (
            group_type
        )

    handle.write(
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


# ============================================================
# Publication-ready figures
# ============================================================

save_boxplot(
    df,
    "outcome",
    "energy_inside_roi",
    [
        "TP",
        "TN",
        "FP",
        "FN",
    ],
    "Grad-CAM Energy Inside ROI by Prediction Outcome",
    "Energy Inside ROI (%)",
    "energy_inside_roi_by_outcome_boxplot.png",
    percent=True,
)

save_boxplot(
    df,
    "outcome",
    "inside_outside_ratio",
    [
        "TP",
        "TN",
        "FP",
        "FN",
    ],
    "Inside-to-Outside Grad-CAM Activation Ratio by Prediction Outcome",
    "Inside / Outside Activation Ratio",
    "activation_ratio_by_outcome_boxplot.png",
    log_scale=True,
)

save_boxplot(
    df,
    "lesion_group",
    "energy_inside_roi",
    [
        "mass",
        "calcification",
        "mixed",
    ],
    "Grad-CAM Energy Inside ROI by Lesion Type",
    "Energy Inside ROI (%)",
    "energy_inside_roi_by_lesion_group_boxplot.png",
    percent=True,
)

save_boxplot(
    df,
    "lesion_group",
    "inside_outside_ratio",
    [
        "mass",
        "calcification",
        "mixed",
    ],
    "Inside-to-Outside Grad-CAM Activation Ratio by Lesion Type",
    "Inside / Outside Activation Ratio",
    "activation_ratio_by_lesion_group_boxplot.png",
    log_scale=True,
)

save_boxplot(
    df,
    "roi_size_group",
    "energy_inside_roi",
    [
        "small",
        "medium",
        "large",
    ],
    "Grad-CAM Energy Inside ROI by ROI Size",
    "Energy Inside ROI (%)",
    "energy_inside_roi_by_roi_size_boxplot.png",
    percent=True,
)

save_boxplot(
    df,
    "roi_size_group",
    "inside_outside_ratio",
    [
        "small",
        "medium",
        "large",
    ],
    "Inside-to-Outside Grad-CAM Activation Ratio by ROI Size",
    "Inside / Outside Activation Ratio",
    "activation_ratio_by_roi_size_boxplot.png",
    log_scale=True,
)


# Pointing-game subgroup charts.
save_pointing_game_bar_chart(
    by_outcome,
    "Pointing-Game Accuracy by Prediction Outcome",
    "pointing_game_by_outcome.png",
)

save_pointing_game_bar_chart(
    by_lesion,
    "Pointing-Game Accuracy by Lesion Type",
    "pointing_game_by_lesion_group.png",
)

save_pointing_game_bar_chart(
    by_roi_size,
    "Pointing-Game Accuracy by ROI Size",
    "pointing_game_by_roi_size.png",
)


# ============================================================
# Print clean summary
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    180
)

print(
    "\nClean subgroup summary:"
)

print(
    clean_table.to_string(
        index=False,
        formatters={
            "Pointing-game accuracy (%)":
                lambda x: f"{x:.1f}",
            "Median energy inside ROI (%)":
                lambda x: f"{x:.3f}",
            "Median activation ratio":
                lambda x: f"{x:.3f}",
            "Median IoU, top 10% (%)":
                lambda x: f"{x:.3f}",
        },
    )
)


# ============================================================
# Final notes
# ============================================================

print("\n")
print("=" * 72)
print("FINAL XAI SUBGROUP ANALYSIS COMPLETE")
print("=" * 72)

print(
    "\nSaved clean report table:"
)

print(
    (
        TABLES_DIR
        / "xai_subgroup_summary_report.csv"
    ).resolve()
)

print(
    "\nSaved LaTeX table:"
)

print(
    latex_path.resolve()
)

print(
    "\nSaved publication-ready figures:"
)

print(
    FIGURES_DIR.resolve()
)
