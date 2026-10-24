"""Regenerate Figures 3-6 from the already-saved test-set probability arrays,
fixing the macro-AUC legend bug: the legend previously showed sk_auc(all_fpr,
mean_tpr) computed from an interpolated mean ROC curve, which is not the same
quantity as multiclass_auc() (mean of per-class roc_auc_score), the function
that produces Table 2/3/4. The two differ by ~0.001-0.003 because averaging
TPR on an interpolated FPR grid then integrating is not equal to integrating
each class's own ROC curve and then averaging the AUCs.

No retraining and no raw images are needed: this uses the already-committed
final_*_proba.npy arrays and y_test_grouped.npy, the same arrays Table 2's
numbers were computed from (verified by exact reproduction, see session notes).
The plotted curve shape is unchanged; only the number printed in the legend
for the macro-average line is corrected to match the authoritative multiclass_auc
value used everywhere else in the manuscript.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc as sk_auc

from full_dataset_advanced_pipeline import multiclass_auc

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "final_figures"

class_names = [
    "Actin disruptors", "Aurora kinase inhibitors", "DMSO", "DNA damage",
    "DNA replication", "Eg5 inhibitors", "Microtubule destabilizers",
    "Microtubule stabilizers", "Protein degradation", "Protein synthesis",
]
n_classes = len(class_names)

y_test = np.load(DATA_DIR / "y_test_grouped.npy")
present_classes = sorted(np.unique(y_test).tolist())
y_bin = label_binarize(y_test, classes=list(range(n_classes)))

proba = {
    "Logistic Regression": np.load(DATA_DIR / "final_logistic_regression_proba.npy"),
    "Random Forest": np.load(DATA_DIR / "final_random_forest_proba.npy"),
    "CNN (scratch)": np.load(DATA_DIR / "final_cnn_scratch_proba.npy"),
    "ResNet-18 (scratch)": np.load(DATA_DIR / "final_resnet18_scratch_proba.npy"),
    "ResNet-18 (pretrained)": np.load(DATA_DIR / "final_resnet18_pretrained_proba.npy"),
}

# Sanity check: these are the exact arrays Table 2 was built from.
expected_table2_auc = {
    "Logistic Regression": 0.866,
    "Random Forest": 0.904,
    "CNN (scratch)": 0.560,
    "ResNet-18 (scratch)": 0.884,
    "ResNet-18 (pretrained)": 0.959,
}
for name, p in proba.items():
    got = round(multiclass_auc(y_test, p), 3)
    want = expected_table2_auc[name]
    assert abs(got - want) < 0.0015, (
        f"{name}: multiclass_auc on final_*_proba.npy ({got}) does not match "
        f"Table 2 ({want}) -- these are not the arrays Table 2 was computed from, "
        f"do not overwrite figures from this script."
    )
print("Verified: all five saved proba arrays reproduce Table 2 exactly.")


def mean_curve(p):
    all_fpr = np.unique(np.concatenate([
        roc_curve(y_bin[:, c], p[:, c])[0] for c in present_classes
    ]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in present_classes:
        fpr_c, tpr_c, _ = roc_curve(y_bin[:, c], p[:, c])
        mean_tpr += np.interp(all_fpr, fpr_c, tpr_c)
    mean_tpr /= len(present_classes)
    return all_fpr, mean_tpr


def plot_macro(models: dict, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for mi, (name, p) in enumerate(models.items()):
        all_fpr, mean_tpr = mean_curve(p)
        legend_auc = multiclass_auc(y_test, p)  # fixed: matches Table 2 exactly
        ax.plot(all_fpr, mean_tpr, label=f"{name} (macro-OvR AUC={legend_auc:.3f})",
                color=colors[mi], linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"wrote {out_path}")


def plot_per_class(p: np.ndarray, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for ci, c in enumerate(present_classes):
        fpr_c, tpr_c, _ = roc_curve(y_bin[:, c], p[:, c])
        auc_c = sk_auc(fpr_c, tpr_c)  # unaffected by the bug: exact per-class AUC
        ax.plot(fpr_c, tpr_c, color=colors[ci], alpha=0.7, linewidth=1,
                label=f"{class_names[c]} (AUC={auc_c:.3f})")
    all_fpr, mean_tpr = mean_curve(p)
    legend_auc = multiclass_auc(y_test, p)  # fixed: matches Table 2 exactly
    ax.plot(all_fpr, mean_tpr, color="black", linewidth=2.5,
            label=f"macro-average (AUC={legend_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"wrote {out_path}")


plot_macro(
    {"Logistic Regression": proba["Logistic Regression"], "Random Forest": proba["Random Forest"]},
    FIG_DIR / "figure3_roc_feature_based_macro.png",
    "Macro-averaged ROC: Feature-Based Classifiers (6 evaluable classes)",
)
plot_per_class(
    proba["Random Forest"],
    FIG_DIR / "figure3b_roc_random_forest_per_class.png",
    "Random Forest: Per-Class ROC with Macro-Average Overlay",
)
plot_macro(
    {
        "CNN (scratch)": proba["CNN (scratch)"],
        "ResNet-18 (scratch)": proba["ResNet-18 (scratch)"],
        "ResNet-18 (pretrained)": proba["ResNet-18 (pretrained)"],
    },
    FIG_DIR / "figure5_roc_deep_learning_macro.png",
    "Macro-averaged ROC: Deep-Learning Models (6 evaluable classes)",
)
plot_per_class(
    proba["ResNet-18 (pretrained)"],
    FIG_DIR / "figure5b_roc_resnet_pretrained_per_class.png",
    "ResNet-18 (Pretrained): Per-Class ROC with Macro-Average Overlay",
)

print("FIGURES_FIXED")
