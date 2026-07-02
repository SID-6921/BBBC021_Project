import sys, time
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import random, torch
from full_dataset_advanced_pipeline import run_resnet_training, multiclass_auc

df = pd.read_pickle("../data/raw/df_default_rebuilt.pkl")
x_img = np.load("../data/raw/x_img_rebuilt.npy")
idx_train = np.load("../data/raw/idx_train.npy")
idx_val = np.load("../data/raw/idx_val.npy")
idx_test = np.load("../data/raw/idx_test.npy")

class_names = sorted(df["class_name"].unique().tolist())
class_to_idx = {c: i for i, c in enumerate(class_names)}
y = df["class_name"].map(class_to_idx).astype(int).values
n_classes = len(class_names)

x_train, y_train = x_img[idx_train], y[idx_train]
x_val, y_val = x_img[idx_val], y[idx_val]
x_test, y_test = x_img[idx_test], y[idx_test]

results = []
for seed in [1, 7, 42, 123, 999]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    t0 = time.time()
    res, _ = run_resnet_training(False, x_train, y_train, x_val, y_val, x_test, y_test, n_classes=n_classes)
    acc = (res.proba_test.argmax(1) == y_test).mean()
    auc = multiclass_auc(y_test, res.proba_test)
    elapsed = time.time() - t0
    print(f"seed={seed} acc={acc:.4f} auc={auc:.4f} time={elapsed:.1f}s", flush=True)
    results.append((seed, acc, auc))
    np.save(f"../data/raw/resnet_scratch_prob_seed{seed}.npy", res.proba_test)

print("RETRY_DONE", flush=True)
print(results, flush=True)
