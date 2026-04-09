from __future__ import annotations

from pathlib import Path
import json
import re
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import load_metadata, load_multichannel_image
from detect import detect_spots
from features import compute_features
from phase_pipeline import ensure_binary_groups
from preprocess import clean_image, normalize_image, resize_image

ZIP_URLS = [
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22123.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22141.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week2_24121.zip",
]
CSV_URL = "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_image.csv"
CONFIGS = {
    "sensitive": {"min_area": 8, "max_area": 1800, "min_mean_intensity": 35.0, "adaptive_block_size": 31, "adaptive_c": -3},
    "default": {"min_area": 10, "max_area": 1800, "min_mean_intensity": 45.0, "adaptive_block_size": 35, "adaptive_c": -4},
    "conservative": {"min_area": 14, "max_area": 1600, "min_mean_intensity": 55.0, "adaptive_block_size": 39, "adaptive_c": -5},
}
FEATURE_COLS = [
    "spot_count",
    "mean_intensity",
    "total_intensity",
    "intensity_variance",
    "area_covered_ratio",
    "density_spots_per_10k_px",
    "spot_area_mean",
    "spot_area_std",
    "spot_area_median",
    "spot_area_q25",
    "spot_area_q75",
    "small_spot_fraction",
    "medium_spot_fraction",
    "large_spot_fraction",
]


def ensure_data(root: Path) -> tuple[Path, Path]:
    raw_dir = root / "data" / "raw"
    images_dir = raw_dir / "images"
    meta_dir = raw_dir / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    for url in ZIP_URLS:
        zip_path = raw_dir / Path(url).name
        if not zip_path.exists():
            urllib.request.urlretrieve(url, zip_path)
        marker = Path(url).stem.replace("BBBC021_v1_images_", "")
        if not any(images_dir.rglob(f"*{marker}*")):
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(images_dir)

    csv_path = meta_dir / "BBBC021_v1_image.csv"
    if not csv_path.exists():
        urllib.request.urlretrieve(CSV_URL, csv_path)

    return images_dir, csv_path


def infer_batch_name(row: pd.Series) -> str:
    for col in row.index:
        if "pathname" not in col.lower() or pd.isna(row[col]):
            continue
        match = re.search(r"Week\d+_\d+", str(row[col]))
        if match:
            return match.group(0)
    table_number = row.get("TableNumber")
    return f"Table_{table_number}"


def get_batch_rows(meta: pd.DataFrame, batch_name: str, images_dir: Path, target_n: int) -> list[tuple[int, pd.Series]]:
    rows: list[tuple[int, pd.Series]] = []
    batch_df = meta[meta["batch_name"] == batch_name]
    channel_cols = [c for c in meta.columns if "filename" in c.lower()]

    for idx, row in batch_df.iterrows():
        try:
            _img, _channels = load_multichannel_image(row, images_dir, channel_cols)
            rows.append((idx, row))
        except Exception:
            continue
        if len(rows) >= target_n:
            break
    return rows


def extract_features_for_config(rows: list[tuple[int, pd.Series]], images_dir: Path, config_name: str, params: dict) -> pd.DataFrame:
    records: list[dict] = []
    for idx, row in rows:
        image_id = f"image_{idx:05d}"
        fused, _ = load_multichannel_image(row, images_dir)
        img = clean_image(normalize_image(resize_image(fused, (512, 512))))
        det = detect_spots(img, **params)
        feat = compute_features(
            image_gray=img,
            mask=det["mask"],
            spot_count=det["spot_count"],
            image_id=image_id,
            group=str(row.get("Image_Metadata_Compound", "unknown")),
            spots=det.get("spots", []),
        )
        feat["batch_name"] = row["batch_name"]
        feat["config_name"] = config_name
        feat["compound"] = str(row.get("Image_Metadata_Compound", "unknown"))
        feat["concentration"] = float(row.get("Image_Metadata_Concentration", np.nan))
        records.append(feat)
    out = pd.DataFrame(records)
    if not out.empty:
        out = ensure_binary_groups(out)
    return out


def evaluate_models(df: pd.DataFrame) -> dict[str, float]:
    x = df[FEATURE_COLS].copy()
    y = (df["group"] == "Group B").astype(int)
    if y.nunique() < 2 or len(df) < 20:
        return {
            "logistic_accuracy": np.nan,
            "logistic_roc_auc": np.nan,
            "rf_accuracy": np.nan,
            "rf_roc_auc": np.nan,
        }

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42, stratify=y)

    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1500, random_state=42)),
    ])
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
    ])

    lr.fit(x_train, y_train)
    rf.fit(x_train, y_train)

    lr_proba = lr.predict_proba(x_test)[:, 1]
    rf_proba = rf.predict_proba(x_test)[:, 1]
    lr_pred = (lr_proba >= 0.5).astype(int)
    rf_pred = (rf_proba >= 0.5).astype(int)

    return {
        "logistic_accuracy": float(accuracy_score(y_test, lr_pred)),
        "logistic_roc_auc": float(roc_auc_score(y_test, lr_proba)),
        "rf_accuracy": float(accuracy_score(y_test, rf_pred)),
        "rf_roc_auc": float(roc_auc_score(y_test, rf_proba)),
    }


def save_figures(all_features: pd.DataFrame, batch_summary: pd.DataFrame, class_df: pd.DataFrame, final_figures: Path) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 5.5))
    sns.boxplot(data=all_features, x="batch_name", y="spot_count", hue="config_name")
    plt.title("Figure 6: Spot Count Robustness Across Batches and Threshold Settings")
    plt.xlabel("Batch")
    plt.ylabel("Spot count")
    plt.tight_layout()
    plt.savefig(final_figures / "figure6_batch_robustness.png", dpi=180)
    plt.close()

    plot_df = batch_summary[["batch_name", "config_name", "mean_spot_count", "cv_spot_count"]].copy()
    melted = plot_df.melt(id_vars=["batch_name", "config_name"], var_name="metric", value_name="value")
    g = sns.catplot(data=melted, x="config_name", y="value", hue="batch_name", col="metric", kind="bar", sharey=False, height=4.5, aspect=1.2)
    g.fig.subplots_adjust(top=0.82)
    g.fig.suptitle("Figure 7: Detection Sensitivity and Stability by Configuration")
    g.savefig(final_figures / "figure7_threshold_sensitivity.png", dpi=180)
    plt.close("all")

    plt.figure(figsize=(8, 5))
    class_long = class_df.melt(id_vars="config_name", value_vars=["logistic_accuracy", "rf_accuracy"], var_name="model", value_name="accuracy")
    sns.barplot(data=class_long, x="config_name", y="accuracy", hue="model")
    plt.ylim(0, 1)
    plt.title("Figure 8: Classification Accuracy by Detection Configuration")
    plt.tight_layout()
    plt.savefig(final_figures / "figure8_robustness_classification.png", dpi=180)
    plt.close()


def run(root: Path, target_per_batch: int = 50) -> None:
    final_figures = root / "final_figures"
    final_tables = root / "final_tables"
    results_summary = root / "results_summary"
    for d in [final_figures, final_tables, results_summary]:
        d.mkdir(parents=True, exist_ok=True)

    images_dir, csv_path = ensure_data(root)
    meta = load_metadata(csv_path)
    meta["batch_name"] = meta.apply(infer_batch_name, axis=1)

    requested_batches = [Path(url).stem.replace("BBBC021_v1_images_", "") for url in ZIP_URLS]
    all_features_frames: list[pd.DataFrame] = []
    batch_summary_rows: list[dict] = []
    class_rows: list[dict] = []

    for config_name, params in CONFIGS.items():
        config_frames: list[pd.DataFrame] = []
        for batch_name in requested_batches:
            rows = get_batch_rows(meta, batch_name, images_dir, target_per_batch)
            features_df = extract_features_for_config(rows, images_dir, config_name, params)
            if features_df.empty:
                continue
            config_frames.append(features_df)
            all_features_frames.append(features_df)

            mean_spot_count = float(features_df["spot_count"].mean())
            std_spot_count = float(features_df["spot_count"].std(ddof=0))
            cv_spot_count = float(std_spot_count / mean_spot_count) if mean_spot_count > 0 else np.nan
            batch_summary_rows.append(
                {
                    "batch_name": batch_name,
                    "config_name": config_name,
                    "n_images": int(len(features_df)),
                    "mean_spot_count": mean_spot_count,
                    "std_spot_count": std_spot_count,
                    "cv_spot_count": cv_spot_count,
                    "mean_density": float(features_df["density_spots_per_10k_px"].mean()),
                    "mean_area_ratio": float(features_df["area_covered_ratio"].mean()),
                    "mean_intensity_variance": float(features_df["intensity_variance"].mean()),
                }
            )

        if config_frames:
            pooled_df = pd.concat(config_frames, ignore_index=True)
            scores = evaluate_models(pooled_df)
            scores["config_name"] = config_name
            class_rows.append(scores)

    all_features = pd.concat(all_features_frames, ignore_index=True)
    batch_summary = pd.DataFrame(batch_summary_rows)
    class_df = pd.DataFrame(class_rows)

    config_summary = (
        batch_summary.groupby("config_name")[["mean_spot_count", "cv_spot_count", "mean_density", "mean_area_ratio"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    all_features.to_csv(final_tables / "robustness_feature_table.csv", index=False)
    batch_summary.to_csv(results_summary / "robustness_batch_summary.csv", index=False)
    class_df.to_csv(results_summary / "robustness_classification.csv", index=False)
    config_summary.to_csv(results_summary / "robustness_config_summary.csv", index=False)

    summary = {
        "batches_tested": requested_batches,
        "target_images_per_batch": target_per_batch,
        "total_images_processed": int(len(all_features)),
        "configs_tested": list(CONFIGS.keys()),
        "best_rf_accuracy_config": None if class_df.empty else class_df.sort_values("rf_accuracy", ascending=False).iloc[0]["config_name"],
    }
    with open(results_summary / "robustness_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_figures(all_features, batch_summary, class_df, final_figures)

    lines = [
        "# Robustness Summary",
        "",
        f"- Batches tested: {', '.join(requested_batches)}",
        f"- Images processed: {len(all_features)}",
        f"- Configurations tested: {', '.join(CONFIGS.keys())}",
        "",
        "## Classification by Configuration",
    ]
    for _, row in class_df.iterrows():
        lines.append(
            f"- {row['config_name']}: LR acc={row['logistic_accuracy']:.4f}, LR AUC={row['logistic_roc_auc']:.4f}, RF acc={row['rf_accuracy']:.4f}, RF AUC={row['rf_roc_auc']:.4f}"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "- final_figures/figure6_batch_robustness.png",
            "- final_figures/figure7_threshold_sensitivity.png",
            "- final_figures/figure8_robustness_classification.png",
            "- final_tables/robustness_feature_table.csv",
            "- results_summary/robustness_batch_summary.csv",
            "- results_summary/robustness_classification.csv",
            "- results_summary/robustness_config_summary.csv",
        ]
    )
    (results_summary / "robustness_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("Completed robustness pipeline")
    print(f"images_processed={len(all_features)}")
    print(class_df.to_string(index=False))


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    run(project_root, target_per_batch=50)
