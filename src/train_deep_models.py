import sys, time
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from full_dataset_advanced_pipeline import (
    set_seed, SEED, run_cnn_training, run_resnet_training, multiclass_auc
)

set_seed(SEED)
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

print("training CNN...", flush=True)
t0 = time.time()
cnn_res = run_cnn_training(x_train, y_train, x_val, y_val, x_test, y_test, n_classes=n_classes)
print("CNN done in", time.time()-t0, "AUC=", multiclass_auc(y_test, cnn_res.proba_test), flush=True)
np.save("../data/raw/cnn_prob.npy", cnn_res.proba_test)

print("training ResNet18 scratch...", flush=True)
t0 = time.time()
res_scratch, _ = run_resnet_training(False, x_train, y_train, x_val, y_val, x_test, y_test, n_classes=n_classes)
print("ResNet18 scratch done in", time.time()-t0, "AUC=", multiclass_auc(y_test, res_scratch.proba_test), flush=True)
np.save("../data/raw/resnet_scratch_prob.npy", res_scratch.proba_test)

print("training ResNet18 pretrained...", flush=True)
t0 = time.time()
res_pre, _ = run_resnet_training(True, x_train, y_train, x_val, y_val, x_test, y_test, n_classes=n_classes)
print("ResNet18 pretrained done in", time.time()-t0, "AUC=", multiclass_auc(y_test, res_pre.proba_test), flush=True)
np.save("../data/raw/resnet_pretrained_prob.npy", res_pre.proba_test)

np.save("../data/raw/y_test_dl.npy", y_test)
print("DEEP_TRAINING_DONE", flush=True)
