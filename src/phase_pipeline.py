from __future__ import annotations

from pathlib import Path
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import assign_group_labels, get_channel_columns, load_metadata, load_multichannel_image
from detect import create_overlay, detect_spots
from features import compute_features
from preprocess import clean_image, normalize_image, resize_image


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "overlay_dir": root / "outputs" / "overlays",
        "metrics_dir": root / "outputs" / "metrics",
        "plots_dir": root / "outputs" / "plots",
        "final_figures": root / "final_figures",
        "final_tables": root / "final_tables",
        "results_summary": root / "results_summary",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def find_available_rows(meta: pd.DataFrame, images_dir: Path, target_n: int) -> list[tuple[int, pd.Series]]:
    channel_cols = get_channel_columns(meta)
    rows: list[tuple[int, pd.Series]] = []
    for idx, row in meta.iterrows():
        try:
            _img, _chs = load_multichannel_image(row, images_dir, channel_cols)
            rows.append((idx, row))
        except Exception:
            continue
        if len(rows) >= target_n:
            break
    return rows


def build_metrics(meta: pd.DataFrame, images_dir: Path, overlay_dir: Path, target_n: int = 220) -> pd.DataFrame:
    sample_rows = find_available_rows(meta, images_dir, target_n)

    records: list[dict] = []
    for i, (idx, row) in enumerate(sample_rows):
        image_id = f"image_{idx:05d}"
        fused, _ = load_multichannel_image(row, images_dir, get_channel_columns(meta))
        img = clean_image(normalize_image(resize_image(fused, (512, 512))))

        det = detect_spots(
            img,
            min_area=10,
            max_area=1800,
            min_mean_intensity=45.0,
            adaptive_block_size=35,
            adaptive_c=-4,
        )

        feat = compute_features(
            image_gray=img,
            mask=det["mask"],
            spot_count=det["spot_count"],
            image_id=image_id,
            group=row["group"],
            spots=det.get("spots", []),
        )
        feat["compound"] = str(row.get("Image_Metadata_Compound", "unknown"))
        feat["concentration"] = float(row.get("Image_Metadata_Concentration", np.nan))
        feat["table_number"] = int(row.get("TableNumber", -1)) if not pd.isna(row.get("TableNumber", np.nan)) else -1
        records.append(feat)

        if i < 12:
            overlay = create_overlay(img, det["contours"], spots=det.get("spots", []))
            cv2.imwrite(str(overlay_dir / f"{image_id}_overlay.png"), overlay)

    return pd.DataFrame(records)


def ensure_binary_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Force usable Group A vs Group B labels from available sample metadata."""
    out = df.copy()

    unique_groups = set(out["group"].dropna().astype(str).unique())
    if unique_groups.issubset({"Group A", "Group B"}) and len(unique_groups) >= 2:
        return out

    if "compound" in out.columns and out["compound"].nunique(dropna=True) >= 2:
        top_compound = out["compound"].value_counts().idxmax()
        out["group"] = np.where(out["compound"] == top_compound, "Group A", "Group B")
        return out

    if "concentration" in out.columns and out["concentration"].nunique(dropna=True) >= 2:
        med = float(out["concentration"].median())
        out["group"] = np.where(out["concentration"] <= med, "Group A", "Group B")
        return out

    med_spots = float(out["spot_count"].median())
    out["group"] = np.where(out["spot_count"] <= med_spots, "Group A", "Group B")
    return out


def save_feature_plots(df: pd.DataFrame, final_figures_dir: Path) -> list[str]:
    sns.set_theme(style="whitegrid")

    feature_names = [
        "spot_count",
        "mean_intensity",
        "density_spots_per_10k_px",
    ]
    saved = []

    for metric in feature_names:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="group", y=metric, fill=False)
        sns.stripplot(data=df, x="group", y=metric, alpha=0.45, size=3)
        plt.title(f"{metric} comparison by group")
        plt.tight_layout()
        out = final_figures_dir / f"figure3_{metric}_boxplot.png"
        plt.savefig(out, dpi=180)
        plt.close()
        saved.append(out.name)

    return saved


def run_classification(df: pd.DataFrame, final_figures_dir: Path, results_summary_dir: Path) -> dict:
    feature_cols = [
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

    x = df[feature_cols].copy()
    y = (df["group"] == "Group B").astype(int)

    if y.nunique() < 2:
        raise ValueError("Classification requires at least two classes in 'group'.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    lr = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1500, random_state=42)),
        ]
    )
    rf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
        ]
    )

    lr.fit(x_train, y_train)
    rf.fit(x_train, y_train)

    lr_proba = lr.predict_proba(x_test)[:, 1]
    rf_proba = rf.predict_proba(x_test)[:, 1]

    lr_pred = (lr_proba >= 0.5).astype(int)
    rf_pred = (rf_proba >= 0.5).astype(int)

    results = {
        "logistic_regression": {
            "accuracy": float(accuracy_score(y_test, lr_pred)),
            "roc_auc": float(roc_auc_score(y_test, lr_proba)),
        },
        "random_forest": {
            "accuracy": float(accuracy_score(y_test, rf_pred)),
            "roc_auc": float(roc_auc_score(y_test, rf_proba)),
        },
    }

    plt.figure(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y_test, lr_proba, name="Logistic Regression")
    RocCurveDisplay.from_predictions(y_test, rf_proba, name="Random Forest")
    plt.title("Figure 4: ROC Curves")
    plt.tight_layout()
    plt.savefig(final_figures_dir / "figure4_classification_roc.png", dpi=180)
    plt.close()

    lr_model = lr.named_steps["model"]
    lr_coef = pd.Series(np.abs(lr_model.coef_[0]), index=feature_cols).sort_values(ascending=False)
    rf_model = rf.named_steps["model"]
    rf_imp = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    imp_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "logistic_abs_coef": [float(lr_coef.get(f, 0.0)) for f in feature_cols],
            "rf_importance": [float(rf_imp.get(f, 0.0)) for f in feature_cols],
        }
    )
    imp_df.to_csv(results_summary_dir / "feature_importance.csv", index=False)

    top = imp_df.sort_values("rf_importance", ascending=False).head(10)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=top, x="rf_importance", y="feature", orient="h")
    plt.title("Figure 4: Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(final_figures_dir / "figure4_feature_importance.png", dpi=180)
    plt.close()

    with open(results_summary_dir / "classification_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def run_unsupervised(df: pd.DataFrame, final_figures_dir: Path, results_summary_dir: Path) -> None:
    feature_cols = [
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

    x = df[feature_cols].copy()
    x = SimpleImputer(strategy="median").fit_transform(x)
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2, random_state=42)
    x_pca = pca.fit_transform(x)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=25)
    clusters = kmeans.fit_predict(x)

    pca_df = pd.DataFrame(
        {
            "PC1": x_pca[:, 0],
            "PC2": x_pca[:, 1],
            "group": df["group"].values,
            "cluster": clusters,
        }
    )
    pca_df.to_csv(results_summary_dir / "pca_clusters.csv", index=False)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="group", style="cluster", s=60)
    plt.title("Figure 5: PCA Projection with KMeans Clusters")
    plt.tight_layout()
    plt.savefig(final_figures_dir / "figure5_pca_clustering.png", dpi=180)
    plt.close()


def save_workflow_figure(final_figures_dir: Path) -> None:
    plt.figure(figsize=(11, 4.5))
    plt.axis("off")

    boxes = [
        (0.05, 0.45, "Input Images + Metadata"),
        (0.25, 0.45, "Preprocessing\nResize + Normalize + Clean"),
        (0.47, 0.45, "Robust Detection\nAdaptive Threshold + Morphology"),
        (0.70, 0.45, "Feature Extraction\nIntensity + Size + Density"),
        (0.90, 0.45, "Modeling\nLR + RF + PCA/KMeans"),
    ]

    for x, y, label in boxes:
        plt.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f4f7fb", "edgecolor": "#0c4a6e", "linewidth": 1.2},
        )

    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + 0.06
        x1 = boxes[i + 1][0] - 0.08
        y = boxes[i][1]
        plt.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops={"arrowstyle": "->", "linewidth": 1.5, "color": "#0c4a6e"})

    plt.title("Figure 1: End-to-End Analysis Pipeline", fontsize=13)
    plt.tight_layout()
    plt.savefig(final_figures_dir / "figure1_pipeline_workflow.png", dpi=180)
    plt.close()


def save_before_after_figure(overlay_dir: Path, final_figures_dir: Path) -> None:
    overlays = sorted(overlay_dir.glob("*_overlay.png"))[:6]
    if not overlays:
        return

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for ax, path in zip(axes, overlays):
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(path.stem)
        ax.axis("off")

    for ax in axes[len(overlays) :]:
        ax.axis("off")

    plt.suptitle("Figure 2: Sample Detection Overlays", y=0.98)
    plt.tight_layout()
    plt.savefig(final_figures_dir / "figure2_detection_samples.png", dpi=180)
    plt.close()


def write_summary(results: dict, df: pd.DataFrame, summary_dir: Path) -> None:
    lines = [
        "# Results Summary",
        "",
        f"- Images processed: {len(df)}",
        f"- Group A count: {(df['group'] == 'Group A').sum()}",
        f"- Group B count: {(df['group'] == 'Group B').sum()}",
        "",
        "## Classification",
        f"- Logistic Regression accuracy: {results['logistic_regression']['accuracy']:.4f}",
        f"- Logistic Regression ROC AUC: {results['logistic_regression']['roc_auc']:.4f}",
        f"- Random Forest accuracy: {results['random_forest']['accuracy']:.4f}",
        f"- Random Forest ROC AUC: {results['random_forest']['roc_auc']:.4f}",
        "",
        "## Key Outputs",
        "- final_figures/figure1_pipeline_workflow.png",
        "- final_figures/figure2_detection_samples.png",
        "- final_figures/figure3_spot_count_boxplot.png",
        "- final_figures/figure3_mean_intensity_boxplot.png",
        "- final_figures/figure3_density_spots_per_10k_px_boxplot.png",
        "- final_figures/figure4_classification_roc.png",
        "- final_figures/figure4_feature_importance.png",
        "- final_figures/figure5_pca_clustering.png",
        "- final_tables/image_feature_table.csv",
        "- results_summary/classification_metrics.json",
        "- results_summary/pca_clusters.csv",
    ]
    (summary_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(root: Path, target_n: int = 220) -> None:
    dirs = ensure_dirs(root)

    meta_path = root / "data" / "raw" / "metadata" / "BBBC021_v1_image.csv"
    images_dir = root / "data" / "raw" / "images"

    if not meta_path.exists():
        raise FileNotFoundError("Metadata CSV not found. Run notebooks/01_data_exploration.ipynb first.")

    meta = assign_group_labels(load_metadata(meta_path))

    metrics_df = build_metrics(meta, images_dir, dirs["overlay_dir"], target_n=target_n)
    if metrics_df.empty:
        raise RuntimeError("No images were processed. Check downloaded images and metadata path mappings.")

    metrics_df = ensure_binary_groups(metrics_df)

    metrics_df.to_csv(dirs["metrics_dir"] / "image_metrics_expanded.csv", index=False)
    metrics_df.to_csv(dirs["final_tables"] / "image_feature_table.csv", index=False)

    feature_plot_files = save_feature_plots(metrics_df, dirs["final_figures"])
    results = run_classification(metrics_df, dirs["final_figures"], dirs["results_summary"])
    run_unsupervised(metrics_df, dirs["final_figures"], dirs["results_summary"])
    save_workflow_figure(dirs["final_figures"])
    save_before_after_figure(dirs["overlay_dir"], dirs["final_figures"])
    write_summary(results, metrics_df, dirs["results_summary"])

    print("Completed phase pipeline")
    print("images_processed=", len(metrics_df))
    print("feature_plots=", feature_plot_files)
    print("lr_accuracy=", f"{results['logistic_regression']['accuracy']:.4f}")
    print("rf_accuracy=", f"{results['random_forest']['accuracy']:.4f}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    run(project_root, target_n=220)
