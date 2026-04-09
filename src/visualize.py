from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRIC_COLUMNS = [
    "spot_count",
    "avg_brightness",
    "total_intensity",
    "area_covered_px",
    "area_covered_ratio",
]


def save_group_plots(df: pd.DataFrame, output_dir: str | Path, group_col: str = "group") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in METRIC_COLUMNS:
        if metric not in df.columns:
            continue

        plt.figure(figsize=(8, 5))
        for group_name, group_df in df.groupby(group_col):
            plt.scatter([group_name] * len(group_df), group_df[metric], alpha=0.7, label=group_name)

        means = df.groupby(group_col)[metric].mean()
        plt.plot(means.index, means.values, color="black", linewidth=2, marker="o", label="group mean")
        plt.title(f"{metric} by {group_col}")
        plt.ylabel(metric)
        plt.xlabel(group_col)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_by_{group_col}.png", dpi=150)
        plt.close()
