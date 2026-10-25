"""Regenerate every figure that can be reproduced from already-saved/cached
data (no raw images, no retraining) at 300 dpi, then export each figure as
its own file named by citation order for journal submission.

Regenerable from cached data: Figures 3, 4, 5, 6 (ROC, from saved test proba
arrays), 7 (seed distribution, from multiseed_resnet_results.csv), 11
(calibration, from saved test proba arrays), 12 (feature ablation, from the
per-channel feature table), 13-14 (PCA batch effects, from the per-channel
feature table), and Supplementary S1-S3 (confusion matrices) and S4
(correlation matrix).

NOT regenerable here (left at their existing resolution, flagged at the end):
Figure 1 (pipeline diagram) and Figure 2 (detection overlays) have no
generating script in this repo; Figures 8, 9, 10 (training/validation loss
curves) need per-epoch loss history that was never cached, only the raw
BBBC021 images + full retraining would reproduce them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc as sk_auc

from full_dataset_advanced_pipeline import (
    SEED, set_seed, split_indices_grouped, fit_classical_models,
    FEATURE_COLS_PER_CHANNEL, save_confusion_matrix, feature_ablation,
    save_feature_correlation, save_calibration_plot, save_pca_batch_effects,
    multiclass_auc,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
RES_DIR = ROOT / "results_summary"
FIG_DIR = ROOT / "final_figures"
EXPORT_DIR = ROOT / "final_figures" / "export_highres"
EXPORT_DIR.mkdir(exist_ok=True)
DPI = 300

df = pd.read_csv(ROOT / "final_tables" / "full_dataset_features_1500plus.csv")
class_names = sorted(df["class_name"].unique().tolist())
class_to_idx = {c: i for i, c in enumerate(class_names)}
y = df["class_name"].map(class_to_idx).astype(int).values
n_classes = len(class_names)

set_seed(SEED)
idx_train, idx_val, idx_test = split_indices_grouped(df, y, group_col="compound", seed=SEED)
y_test = y[idx_test]
assert np.array_equal(y_test, np.load(DATA_DIR / "y_test_grouped.npy")), "split mismatch"
present_classes = sorted(np.unique(y_test).tolist())
y_bin = label_binarize(y_test, classes=list(range(n_classes)))

proba = {
    "Logistic Regression": np.load(DATA_DIR / "final_logistic_regression_proba.npy"),
    "Random Forest": np.load(DATA_DIR / "final_random_forest_proba.npy"),
    "CNN (scratch)": np.load(DATA_DIR / "final_cnn_scratch_proba.npy"),
    "ResNet-18 (scratch)": np.load(DATA_DIR / "final_resnet18_scratch_proba.npy"),
    "ResNet-18 (pretrained)": np.load(DATA_DIR / "final_resnet18_pretrained_proba.npy"),
}
for name, want in {"Logistic Regression": 0.866, "Random Forest": 0.904, "CNN (scratch)": 0.560,
                    "ResNet-18 (scratch)": 0.884, "ResNet-18 (pretrained)": 0.959}.items():
    got = round(multiclass_auc(y_test, proba[name]), 3)
    assert abs(got - want) < 0.0015, f"{name} proba array does not match Table 2 ({got} vs {want})"


def mean_curve(p):
    all_fpr = np.unique(np.concatenate([roc_curve(y_bin[:, c], p[:, c])[0] for c in present_classes]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in present_classes:
        fpr_c, tpr_c, _ = roc_curve(y_bin[:, c], p[:, c])
        mean_tpr += np.interp(all_fpr, fpr_c, tpr_c)
    mean_tpr /= len(present_classes)
    return all_fpr, mean_tpr


def plot_macro(models, out_path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for mi, (name, p) in enumerate(models.items()):
        all_fpr, mean_tpr = mean_curve(p)
        legend_auc = multiclass_auc(y_test, p)
        ax.plot(all_fpr, mean_tpr, label=f"{name} (macro-OvR AUC={legend_auc:.3f})", color=colors[mi], linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig(out_path, dpi=DPI); plt.close()


def plot_per_class(p, out_path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for ci, c in enumerate(present_classes):
        fpr_c, tpr_c, _ = roc_curve(y_bin[:, c], p[:, c])
        auc_c = sk_auc(fpr_c, tpr_c)
        ax.plot(fpr_c, tpr_c, color=colors[ci], alpha=0.7, linewidth=1, label=f"{class_names[c]} (AUC={auc_c:.3f})")
    all_fpr, mean_tpr = mean_curve(p)
    legend_auc = multiclass_auc(y_test, p)
    ax.plot(all_fpr, mean_tpr, color="black", linewidth=2.5, label=f"macro-average (AUC={legend_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout(); plt.savefig(out_path, dpi=DPI); plt.close()


# Figures 3-6 (ROC)
plot_macro({"Logistic Regression": proba["Logistic Regression"], "Random Forest": proba["Random Forest"]},
           EXPORT_DIR / "Figure_03.png", "Macro-averaged ROC: Feature-Based Classifiers (6 evaluable classes)")
plot_per_class(proba["Random Forest"], EXPORT_DIR / "Figure_04.png",
               "Random Forest: Per-Class ROC with Macro-Average Overlay")
plot_macro({"CNN (scratch)": proba["CNN (scratch)"], "ResNet-18 (scratch)": proba["ResNet-18 (scratch)"],
            "ResNet-18 (pretrained)": proba["ResNet-18 (pretrained)"]},
           EXPORT_DIR / "Figure_05.png", "Macro-averaged ROC: Deep-Learning Models (6 evaluable classes)")
plot_per_class(proba["ResNet-18 (pretrained)"], EXPORT_DIR / "Figure_06.png",
               "ResNet-18 (Pretrained): Per-Class ROC with Macro-Average Overlay")
print("Figures 3-6 exported at 300 dpi.")

# Figure 7 (seed distribution) from cached per-seed results
seeds_df = pd.read_csv(RES_DIR / "multiseed_resnet_results.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("ResNet-18 performance across 5 random seeds (1, 7, 42, 123, 999)")
for ax, metric, ylabel in [(axes[0], "accuracy", "Accuracy"), (axes[1], "macro_auc", "Macro-OvR AUC")]:
    for mi, model in enumerate(["resnet18_scratch", "resnet18_pretrained"]):
        sub = seeds_df[seeds_df["model"] == model]
        color = "indianred" if model == "resnet18_scratch" else "mediumseagreen"
        ax.scatter([mi] * len(sub), sub[metric], color=color, s=80, zorder=3,
                   label="ResNet-18 (scratch)" if model == "resnet18_scratch" else "ResNet-18 (pretrained)")
        ax.hlines(sub[metric].mean(), mi - 0.2, mi + 0.2, color="black", linewidth=2, zorder=4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["ResNet-18 (scratch)", "ResNet-18 (pretrained)"])
    ax.set_ylabel(ylabel); ax.set_title(ylabel)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles + [plt.Line2D([0], [0], color="black", lw=2)], labels + ["mean"],
           loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.savefig(EXPORT_DIR / "Figure_07.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Figure 7 exported at 300 dpi (from multiseed_resnet_results.csv, not retrained).")

# Figure 11 (calibration) from saved proba arrays
calib_probs = {
    "logistic_regression": proba["Logistic Regression"], "random_forest": proba["Random Forest"],
    "cnn_scratch": proba["CNN (scratch)"], "resnet18_scratch": proba["ResNet-18 (scratch)"],
    "resnet18_pretrained": proba["ResNet-18 (pretrained)"],
}
ece_df = save_calibration_plot(y_test, calib_probs, EXPORT_DIR / "Figure_11.png")
print("Figure 11 exported at 300 dpi (ECE check):", ece_df.to_dict("records"))

# Figures 13-14 (PCA batch effects) from the full per-channel feature table
batch_metrics = save_pca_batch_effects(df, EXPORT_DIR / "Figure_13.png", EXPORT_DIR / "Figure_14.png")
print("Figures 13-14 exported (batch metrics check):", batch_metrics)

# Figure 12 (ablation), S1-S3 (confusion matrices), S4 (correlation matrix)
x_train = df.iloc[idx_train][FEATURE_COLS_PER_CHANNEL]
y_train = y[idx_train]
x_test = df.iloc[idx_test][FEATURE_COLS_PER_CHANNEL]
set_seed(SEED)
lr, rf, _, _ = fit_classical_models(x_train, y_train)

ablation_df = feature_ablation(rf, x_train, y_train, x_test, y_test)
plt.figure(figsize=(8, 5))
sns.barplot(data=ablation_df, x="ablation", y="test_auc")
plt.xticks(rotation=30, ha="right"); plt.ylim(0.0, 1.0)
plt.title("Feature Ablation Study (Random Forest)")
plt.tight_layout(); plt.savefig(EXPORT_DIR / "Figure_12.png", dpi=DPI); plt.close()

save_confusion_matrix(y_test, proba["Logistic Regression"].argmax(1), class_names,
                       EXPORT_DIR / "Supplementary_Figure_S1.png", "Confusion Matrix - Logistic Regression")
save_confusion_matrix(y_test, proba["Random Forest"].argmax(1), class_names,
                       EXPORT_DIR / "Supplementary_Figure_S2.png", "Confusion Matrix - Random Forest")
save_confusion_matrix(y_test, proba["ResNet-18 (pretrained)"].argmax(1), class_names,
                       EXPORT_DIR / "Supplementary_Figure_S3.png", "Confusion Matrix - ResNet-18 Pretrained")
save_feature_correlation(df, EXPORT_DIR / "Supplementary_Figure_S4.png")
print("Figure 12 and Supplementary S1-S4 exported at 300 dpi.")

print("DONE. Still need the original artifacts for Figure 1, Figure 2, Figure 8, Figure 9, Figure 10.")
