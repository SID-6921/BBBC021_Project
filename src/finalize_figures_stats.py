import sys, json
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc as sk_auc

from full_dataset_advanced_pipeline import (
    SEED, split_indices_grouped, fit_classical_models, run_cnn_training, run_resnet_training,
    multiclass_auc, paired_bootstrap_auc_test, delong_test_multiclass_macro, set_seed,
    FEATURE_COLS_PER_CHANNEL,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE_DF = ROOT / "data" / "raw" / "df_default_grouped.pkl"
CACHE_IMG = ROOT / "data" / "raw" / "x_img_grouped.npy"
FIG_DIR = ROOT / "final_figures"
RES_DIR = ROOT / "results_summary"

print("Loading cached per-channel dataset...", flush=True)
df_default = pd.read_pickle(CACHE_DF)
x_img = np.load(CACHE_IMG)

class_names = sorted(df_default["class_name"].unique().tolist())
class_to_idx = {c: i for i, c in enumerate(class_names)}
y = df_default["class_name"].map(class_to_idx).astype(int).values
n_classes = len(class_names)

set_seed(SEED)
idx_train, idx_val, idx_test = split_indices_grouped(df_default, y, group_col="compound", seed=SEED)
print(f"n_train={len(idx_train)} n_val={len(idx_val)} n_test={len(idx_test)}", flush=True)

x_train_feat = df_default.iloc[idx_train][FEATURE_COLS_PER_CHANNEL]
y_train = y[idx_train]
x_test_feat = df_default.iloc[idx_test][FEATURE_COLS_PER_CHANNEL]
y_test = y[idx_test]

print("Fitting LR/RF...", flush=True)
lr, rf, _, _ = fit_classical_models(x_train_feat, y_train)
lr_prob = lr.predict_proba(x_test_feat)
rf_prob = rf.predict_proba(x_test_feat)

x_train_img, x_val_img, x_test_img = x_img[idx_train], x_img[idx_val], x_img[idx_test]
y_val = y[idx_val]

print("Training CNN...", flush=True)
cnn_res = run_cnn_training(x_train_img, y_train, x_val_img, y_val, x_test_img, y_test, n_classes=n_classes)
print("Training ResNet scratch...", flush=True)
res_scratch, _ = run_resnet_training(False, x_train_img, y_train, x_val_img, y_val, x_test_img, y_test, n_classes=n_classes)
print("Training ResNet pretrained...", flush=True)
res_pre, _ = run_resnet_training(True, x_train_img, y_train, x_val_img, y_val, x_test_img, y_test, n_classes=n_classes)

model_probs = {
    "logistic_regression": lr_prob,
    "random_forest": rf_prob,
    "cnn_scratch": cnn_res.proba_test,
    "resnet18_scratch": res_scratch.proba_test,
    "resnet18_pretrained": res_pre.proba_test,
}
for name, p in model_probs.items():
    np.save(RES_DIR.parent / "data" / "raw" / f"final_{name}_proba.npy", p)

present_classes = sorted(np.unique(y_test).tolist())
print("present classes:", [class_names[c] for c in present_classes], flush=True)

# ---------------------------------------------------------------------
# Corrected bootstrap stats (fixed p-value formula)
# ---------------------------------------------------------------------
bootstrap_cnn_vs_rf = paired_bootstrap_auc_test(y_test, model_probs["cnn_scratch"], model_probs["random_forest"])
bootstrap_resnet_pre_vs_rf = paired_bootstrap_auc_test(y_test, model_probs["resnet18_pretrained"], model_probs["random_forest"])
delong_cnn_vs_rf = delong_test_multiclass_macro(y_test, model_probs["cnn_scratch"], model_probs["random_forest"])
delong_resnet_pre_vs_rf = delong_test_multiclass_macro(y_test, model_probs["resnet18_pretrained"], model_probs["random_forest"])

with open(RES_DIR / "bootstrap_auc_tests.json", "w", encoding="utf-8") as f:
    json.dump({
        "cnn_scratch_vs_random_forest": bootstrap_cnn_vs_rf,
        "resnet18_pretrained_vs_random_forest": bootstrap_resnet_pre_vs_rf,
    }, f, indent=2)
with open(RES_DIR / "advanced_delong_tests.json", "w", encoding="utf-8") as f:
    json.dump({
        "cnn_scratch_vs_random_forest": delong_cnn_vs_rf,
        "resnet18_pretrained_vs_random_forest": delong_resnet_pre_vs_rf,
    }, f, indent=2)

print("bootstrap:", json.dumps({"cnn_vs_rf": bootstrap_cnn_vs_rf, "resnet_pre_vs_rf": bootstrap_resnet_pre_vs_rf}, indent=2), flush=True)

# ---------------------------------------------------------------------
# Per-class ROC curves with macro-average overlay (fixes R2 #13)
# ---------------------------------------------------------------------
def plot_roc(models: dict, out_path: Path, title: str):
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for mi, (name, proba) in enumerate(models.items()):
        # macro-average ROC over present classes only
        all_fpr = np.unique(np.concatenate([
            roc_curve(y_bin[:, c], proba[:, c])[0] for c in present_classes
        ]))
        mean_tpr = np.zeros_like(all_fpr)
        for c in present_classes:
            fpr_c, tpr_c, _ = roc_curve(y_bin[:, c], proba[:, c])
            mean_tpr += np.interp(all_fpr, fpr_c, tpr_c)
        mean_tpr /= len(present_classes)
        macro_auc = sk_auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, label=f"{name} (macro-OvR AUC={macro_auc:.3f})",
                color=colors[mi], linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def plot_roc_per_class(proba: np.ndarray, out_path: Path, title: str):
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for ci, c in enumerate(present_classes):
        fpr_c, tpr_c, _ = roc_curve(y_bin[:, c], proba[:, c])
        auc_c = sk_auc(fpr_c, tpr_c)
        ax.plot(fpr_c, tpr_c, color=colors[ci], alpha=0.7, linewidth=1,
                label=f"{class_names[c]} (AUC={auc_c:.3f})")
    all_fpr = np.unique(np.concatenate([roc_curve(y_bin[:, c], proba[:, c])[0] for c in present_classes]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in present_classes:
        fpr_c, tpr_c, _ = roc_curve(y_bin[:, c], proba[:, c])
        mean_tpr += np.interp(all_fpr, fpr_c, tpr_c)
    mean_tpr /= len(present_classes)
    macro_auc = sk_auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, color="black", linewidth=2.5, label=f"macro-average (AUC={macro_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

plot_roc(
    {"Logistic Regression": lr_prob, "Random Forest": rf_prob},
    FIG_DIR / "figure3_roc_feature_based_macro.png",
    "Macro-averaged ROC: Feature-Based Classifiers (6 evaluable classes)",
)
plot_roc(
    {"CNN (scratch)": cnn_res.proba_test, "ResNet-18 (scratch)": res_scratch.proba_test, "ResNet-18 (pretrained)": res_pre.proba_test},
    FIG_DIR / "figure5_roc_deep_learning_macro.png",
    "Macro-averaged ROC: Deep-Learning Models (6 evaluable classes)",
)
plot_roc_per_class(rf_prob, FIG_DIR / "figure3b_roc_random_forest_per_class.png", "Random Forest: Per-Class ROC with Macro-Average Overlay")
plot_roc_per_class(res_pre.proba_test, FIG_DIR / "figure5b_roc_resnet_pretrained_per_class.png", "ResNet-18 (Pretrained): Per-Class ROC with Macro-Average Overlay")

print("FINALIZE_DONE", flush=True)
