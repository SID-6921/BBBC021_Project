import sys, time
sys.path.insert(0, ".")
from pathlib import Path
import pandas as pd
import numpy as np
from full_dataset_advanced_pipeline import (
    set_seed, SEED, ensure_data, load_metadata, attach_moa_labels, collect_rows,
    build_dataset, split_indices, DETECT_CONFIGS
)

root = Path("..")
set_seed(SEED)
t0 = time.time()
images_dir, csv_path, moa_path = ensure_data(root)
print("ensure_data done", time.time()-t0, flush=True)
meta = load_metadata(csv_path)
moa = pd.read_csv(moa_path)
meta = attach_moa_labels(meta, moa)
print("meta shape after attach_moa_labels:", meta.shape, flush=True)

rows = collect_rows(meta, images_dir, target_total=1600)
print("collected rows:", len(rows), flush=True)

t1 = time.time()
df_default, x_img = build_dataset(rows, images_dir, DETECT_CONFIGS["default"], dl_size=224)
print("build_dataset done in", time.time()-t1, "shape:", df_default.shape, x_img.shape, flush=True)

class_names = sorted(df_default["class_name"].unique().tolist())
print("classes:", class_names, flush=True)

df_default.to_pickle("../data/raw/df_default_rebuilt.pkl")
np.save("../data/raw/x_img_rebuilt.npy", x_img)

class_to_idx = {c: i for i, c in enumerate(class_names)}
y = df_default["class_name"].map(class_to_idx).astype(int).values
idx_train, idx_val, idx_test = split_indices(y)
np.save("../data/raw/idx_train.npy", idx_train)
np.save("../data/raw/idx_val.npy", idx_val)
np.save("../data/raw/idx_test.npy", idx_test)

split_df = pd.DataFrame({"image_id": df_default["image_id"], "split": "unused"})
split_df.loc[idx_train, "split"] = "train"
split_df.loc[idx_val, "split"] = "val"
split_df.loc[idx_test, "split"] = "test"

orig = pd.read_csv("../results_summary/main_split_indices.csv")
merged = split_df.merge(orig, on="image_id", suffixes=("_new","_orig"), how="outer", indicator=True)
print("match rate:", (merged["split_new"]==merged["split_orig"]).mean(), flush=True)
print("mismatch count:", (merged["split_new"]!=merged["split_orig"]).sum(), flush=True)
print("only in one:", (merged["_merge"]!="both").sum(), flush=True)
print("REBUILD_DONE", flush=True)
