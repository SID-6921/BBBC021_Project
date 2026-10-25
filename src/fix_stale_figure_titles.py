"""Regenerate the figures whose on-image titles were stale leftovers from an
earlier iteration of the pipeline that used 12 classes (12 most-frequent
compound-treatment labels, see the top12_compounds fallback in
rebuild_dataset.py) or a since-changed figure numbering scheme. The current
manuscript uses 10 MOA classes throughout (Table 1); the title text was never
updated when the class count and figure numbering changed.

No retraining and no raw images are needed: confusion matrices and the
correlation matrix only need the already-saved/reproducible test predictions
and the per-channel feature table, both already verified elsewhere in this
project to reproduce Table 2 exactly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from full_dataset_advanced_pipeline import (
    SEED, set_seed, split_indices_grouped, fit_classical_models,
    FEATURE_COLS_PER_CHANNEL, save_confusion_matrix, feature_ablation,
    save_feature_correlation,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "final_figures"

df = pd.read_csv(ROOT / "final_tables" / "full_dataset_features_1500plus.csv")
class_names = sorted(df["class_name"].unique().tolist())
class_to_idx = {c: i for i, c in enumerate(class_names)}
y = df["class_name"].map(class_to_idx).astype(int).values

set_seed(SEED)
idx_train, idx_val, idx_test = split_indices_grouped(df, y, group_col="compound", seed=SEED)

y_test_saved = np.load(DATA_DIR / "y_test_grouped.npy")
assert np.array_equal(y[idx_test], y_test_saved), "split reproduction mismatch, aborting"
y_test = y_test_saved

lr_prob = np.load(DATA_DIR / "final_logistic_regression_proba.npy")
rf_prob = np.load(DATA_DIR / "final_random_forest_proba.npy")
resnet_pre_prob = np.load(DATA_DIR / "final_resnet18_pretrained_proba.npy")

save_confusion_matrix(
    y_test, lr_prob.argmax(1), class_names,
    FIG_DIR / "figure11_confusion_matrix_lr_12class.png",
    "Confusion Matrix - Logistic Regression",
)
save_confusion_matrix(
    y_test, rf_prob.argmax(1), class_names,
    FIG_DIR / "figure12_confusion_matrix_rf_12class.png",
    "Confusion Matrix - Random Forest",
)
save_confusion_matrix(
    y_test, resnet_pre_prob.argmax(1), class_names,
    FIG_DIR / "figure13_confusion_matrix_resnet_pretrained_12class.png",
    "Confusion Matrix - ResNet-18 Pretrained",
)
print("Confusion matrices regenerated with corrected titles.")

x_train = df.iloc[idx_train][FEATURE_COLS_PER_CHANNEL]
y_train = y[idx_train]
x_test = df.iloc[idx_test][FEATURE_COLS_PER_CHANNEL]
set_seed(SEED)
lr, rf, _, _ = fit_classical_models(x_train, y_train)

import matplotlib.pyplot as plt
import seaborn as sns

ablation_df = feature_ablation(rf, x_train, y_train, x_test, y_test)
plt.figure(figsize=(8, 5))
sns.barplot(data=ablation_df, x="ablation", y="test_auc")
plt.xticks(rotation=30, ha="right")
plt.ylim(0.0, 1.0)
plt.title("Feature Ablation Study (Random Forest)")
plt.tight_layout()
plt.savefig(FIG_DIR / "figure17_feature_ablation.png", dpi=180)
plt.close()
print("Feature ablation figure regenerated with corrected title.")

save_feature_correlation(df, FIG_DIR / "supplementary_figure_s1_feature_correlation.png")
print("Correlation matrix figure regenerated, removed incorrect self-numbering (was 'S1', actually cited as S4).")

print("FIGURE_TITLES_FIXED")
