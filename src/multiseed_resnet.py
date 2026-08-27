import sys, time, random, json
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from full_dataset_advanced_pipeline import (
    SEED, ensure_data, load_metadata, attach_moa_labels, collect_rows, build_dataset,
    split_indices_grouped, run_resnet_training, multiclass_auc, DETECT_CONFIGS,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE_DF = ROOT / "data" / "raw" / "df_default_grouped.pkl"
CACHE_IMG = ROOT / "data" / "raw" / "x_img_grouped.npy"

if CACHE_DF.exists() and CACHE_IMG.exists():
    print("Loading cached dataset...", flush=True)
    df_default = pd.read_pickle(CACHE_DF)
    x_img = np.load(CACHE_IMG)
else:
    print("Building dataset (feature extraction)...", flush=True)
    images_dir, csv_path, moa_path = ensure_data(ROOT)
    meta = load_metadata(csv_path)
    moa = pd.read_csv(moa_path)
    meta = attach_moa_labels(meta, moa)
    rows = collect_rows(meta, images_dir, target_total=1600)
    df_default, x_img = build_dataset(rows, images_dir, DETECT_CONFIGS["default"], dl_size=224)
    df_default.to_pickle(CACHE_DF)
    np.save(CACHE_IMG, x_img)
    print(f"Cached to {CACHE_DF} and {CACHE_IMG}", flush=True)

class_names = sorted(df_default["class_name"].unique().tolist())
class_to_idx = {c: i for i, c in enumerate(class_names)}
y = df_default["class_name"].map(class_to_idx).astype(int).values
n_classes = len(class_names)

# Fixed split (same compound-grouped split for every seed -- we want seed variance in
# model init/training, not in which compounds land in train/val/test).
idx_train, idx_val, idx_test = split_indices_grouped(df_default, y, group_col="compound", seed=SEED)
print(f"n_train={len(idx_train)} n_val={len(idx_val)} n_test={len(idx_test)}", flush=True)

x_train, y_train = x_img[idx_train], y[idx_train]
x_val, y_val = x_img[idx_val], y[idx_val]
x_test, y_test = x_img[idx_test], y[idx_test]

SEEDS = [1, 7, 42, 123, 999]
results = []

for seed in SEEDS:
    for pretrained in [False, True]:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        t0 = time.time()
        res, _ = run_resnet_training(pretrained, x_train, y_train, x_val, y_val, x_test, y_test, n_classes=n_classes)
        elapsed = time.time() - t0
        acc = float((res.pred_test == y_test).mean())
        auc = multiclass_auc(y_test, res.proba_test)
        name = "resnet18_pretrained" if pretrained else "resnet18_scratch"
        pred_classes = pd.Series(res.pred_test).map(lambda i: class_names[i]).value_counts().to_dict()
        print(f"seed={seed} model={name} acc={acc:.4f} auc={auc:.4f} time={elapsed:.1f}s pred_top3={dict(list(pred_classes.items())[:3])}", flush=True)
        results.append({"seed": seed, "model": name, "accuracy": acc, "macro_auc": auc, "train_time_sec": elapsed})
        np.save(ROOT / "data" / "raw" / f"{name}_prob_seed{seed}.npy", res.proba_test)

results_df = pd.DataFrame(results)
results_df.to_csv(ROOT / "results_summary" / "multiseed_resnet_results.csv", index=False)

summary = results_df.groupby("model")[["accuracy", "macro_auc"]].agg(["mean", "std", "min", "max"])
print(summary, flush=True)
summary.to_csv(ROOT / "results_summary" / "multiseed_resnet_summary.csv")

np.save(ROOT / "data" / "raw" / "y_test_grouped.npy", y_test)
with open(ROOT / "results_summary" / "multiseed_class_names.json", "w") as f:
    json.dump(class_names, f)

print("MULTISEED_DONE", flush=True)
