from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import random
import re
import time
import tracemalloc
import urllib.request
import zipfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import kruskal, norm, spearmanr
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from data_loader import load_metadata, load_multichannel_image
from detect import detect_spots
from features import compute_features
from phase_pipeline import ensure_binary_groups
from preprocess import clean_image, normalize_image, resize_image

SEED = 42
DL_IMAGE_SIZE = 96
BATCH_SIZE = 32
CNN_EPOCHS = 4
RESNET_EPOCHS = 3
LEARNING_RATE = 1e-3

FULL_ZIP_URLS = [
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22123.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22141.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22161.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22361.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22381.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22401.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week2_24121.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week2_24141.zip",
]
CSV_URL = "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_image.csv"

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


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_data(root: Path) -> tuple[Path, Path]:
    raw_dir = root / "data" / "raw"
    images_dir = raw_dir / "images"
    meta_dir = raw_dir / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    for url in FULL_ZIP_URLS:
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


def collect_rows(meta: pd.DataFrame, images_dir: Path, target_total: int = 1600) -> list[tuple[int, pd.Series]]:
    requested_batches = [Path(url).stem.replace("BBBC021_v1_images_", "") for url in FULL_ZIP_URLS]
    channel_cols = [c for c in meta.columns if "filename" in c.lower()]

    rows: list[tuple[int, pd.Series]] = []
    per_batch_target = int(math.ceil(target_total / max(1, len(requested_batches))))

    for batch in requested_batches:
        subset = meta[meta["batch_name"] == batch]
        added = 0
        for idx, row in subset.iterrows():
            try:
                _img, _chs = load_multichannel_image(row, images_dir, channel_cols)
                rows.append((idx, row))
                added += 1
            except Exception:
                continue
            if added >= per_batch_target or len(rows) >= target_total:
                break
        if len(rows) >= target_total:
            break

    return rows


def build_dataset(rows: list[tuple[int, pd.Series]], images_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    records: list[dict] = []
    images: list[np.ndarray] = []

    for idx, row in rows:
        image_id = f"image_{idx:05d}"
        fused, _ = load_multichannel_image(row, images_dir)

        img_detect = clean_image(normalize_image(resize_image(fused, (256, 256))))
        det = detect_spots(
            img_detect,
            min_area=10,
            max_area=1800,
            min_mean_intensity=45.0,
            adaptive_block_size=35,
            adaptive_c=-4,
        )

        feat = compute_features(
            image_gray=img_detect,
            mask=det["mask"],
            spot_count=det["spot_count"],
            image_id=image_id,
            group=str(row.get("Image_Metadata_Compound", "unknown")),
            spots=det.get("spots", []),
        )
        feat["compound"] = str(row.get("Image_Metadata_Compound", "unknown"))
        feat["concentration"] = float(row.get("Image_Metadata_Concentration", np.nan))
        feat["batch_name"] = infer_batch_name(row)
        records.append(feat)

        img_dl = clean_image(normalize_image(resize_image(fused, (DL_IMAGE_SIZE, DL_IMAGE_SIZE))))
        images.append(img_dl.astype(np.float32) / 255.0)

    df = pd.DataFrame(records)
    df = ensure_binary_groups(df)
    x_img = np.stack(images, axis=0)
    return df, x_img


@dataclass
class TorchResult:
    proba_test: np.ndarray
    train_time_sec: float
    peak_mem_mb: float
    used_pretrained: bool


class GrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ResNet18Binary(nn.Module):
    def __init__(self):
        super().__init__()
        used_pretrained = True
        try:
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            base = models.resnet18(weights=None)
            used_pretrained = False

        for p in base.parameters():
            p.requires_grad = False
        base.fc = nn.Linear(base.fc.in_features, 1)
        self.model = base
        self.used_pretrained = used_pretrained

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def train_torch_binary(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int, lr: float, device: torch.device) -> None:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for xb, _ in val_loader:
                _ = model(xb.to(device))


def predict_proba_torch(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out: list[float] = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            proba = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            out.extend(proba.tolist())
    return np.array(out, dtype=np.float32)


def select_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    best_t = 0.5
    best_acc = -1.0
    for t in np.linspace(0.05, 0.95, 19):
        pred = (y_proba >= t).astype(np.int64)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc = float(acc)
            best_t = float(t)
    return best_t


def fit_classical_models(x_train: pd.DataFrame, y_train: np.ndarray):
    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=350, random_state=SEED, class_weight="balanced")),
    ])

    timings = {}
    mems = {}

    tracemalloc.start()
    t0 = time.perf_counter()
    lr.fit(x_train, y_train)
    timings["logistic_regression"] = time.perf_counter() - t0
    mems["logistic_regression"] = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()

    tracemalloc.start()
    t0 = time.perf_counter()
    rf.fit(x_train, y_train)
    timings["random_forest"] = time.perf_counter() - t0
    mems["random_forest"] = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()

    return lr, rf, timings, mems


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(y_true[mask]))
        bin_conf = float(np.mean(y_prob[mask]))
        ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
    return float(ece)


def per_class_auc(y_true: np.ndarray, y_prob_pos: np.ndarray) -> dict[str, float]:
    return {
        "class_GroupA_auc": float(roc_auc_score(1 - y_true, 1 - y_prob_pos)),
        "class_GroupB_auc": float(roc_auc_score(y_true, y_prob_pos)),
    }


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n, dtype=np.float64)
    out[j] = t
    return out


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=np.float64)
    ty = np.empty((k, n), dtype=np.float64)
    tz = np.empty((k, m + n), dtype=np.float64)

    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(predictions_sorted_transposed[r])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    s = sx / m + sy / n
    return aucs, s


def delong_test(y_true: np.ndarray, pred_one: np.ndarray, pred_two: np.ndarray) -> dict[str, float]:
    order = np.argsort(-y_true)
    label_1_count = int(np.sum(y_true))
    preds = np.vstack([pred_one, pred_two])[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = float(diff / np.sqrt(max(var, 1e-12)))
    p_value = float(2 * norm.sf(abs(z)))
    return {
        "auc_model_1": float(aucs[0]),
        "auc_model_2": float(aucs[1]),
        "auc_diff": float(diff),
        "z_stat": z,
        "p_value": p_value,
    }


def batch_normalize_features(df: pd.DataFrame, feature_cols: list[str], batch_col: str) -> pd.DataFrame:
    out = df.copy()
    out[feature_cols] = out[feature_cols].astype(np.float64)
    for b, bdf in out.groupby(batch_col):
        idx = bdf.index
        vals = bdf[feature_cols]
        mu = vals.mean()
        sigma = vals.std(ddof=0).replace(0, 1.0)
        out.loc[idx, feature_cols] = (vals - mu) / sigma
    return out


def save_pca_batch_effects(df: pd.DataFrame, feature_cols: list[str], out_path_before: Path, out_path_after: Path) -> None:
    x_before = StandardScaler().fit_transform(df[feature_cols])
    pca_before = PCA(n_components=2, random_state=SEED).fit_transform(x_before)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_before[:, 0], y=pca_before[:, 1], hue=df["batch_name"], s=30)
    plt.title("PCA Batch Effects Before Batch-Normalization")
    plt.tight_layout()
    plt.savefig(out_path_before, dpi=180)
    plt.close()

    df_after = batch_normalize_features(df, feature_cols, "batch_name")
    x_after = StandardScaler().fit_transform(df_after[feature_cols])
    pca_after = PCA(n_components=2, random_state=SEED).fit_transform(x_after)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_after[:, 0], y=pca_after[:, 1], hue=df_after["batch_name"], s=30)
    plt.title("PCA Batch Effects After Batch-Normalization")
    plt.tight_layout()
    plt.savefig(out_path_after, dpi=180)
    plt.close()


def nested_cv_assessment(x_train: pd.DataFrame, y_train: np.ndarray) -> pd.DataFrame:
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    lr_grid = {
        "model__C": [0.1, 1.0, 5.0],
    }
    rf_grid = {
        "model__n_estimators": [200, 350],
        "model__max_depth": [None, 8, 16],
    }

    lr_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])
    rf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(random_state=SEED, class_weight="balanced")),
    ])

    rows = []
    for name, pipe, grid in [
        ("logistic_regression", lr_pipe, lr_grid),
        ("random_forest", rf_pipe, rf_grid),
    ]:
        fold_scores = []
        for tr_idx, te_idx in outer_cv.split(x_train, y_train):
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
            gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner_cv, n_jobs=-1)
            gs.fit(x_train.iloc[tr_idx], y_train[tr_idx])
            proba = gs.predict_proba(x_train.iloc[te_idx])[:, 1]
            fold_scores.append(roc_auc_score(y_train[te_idx], proba))
        rows.append(
            {
                "model": name,
                "nested_cv_auc_mean": float(np.mean(fold_scores)),
                "nested_cv_auc_std": float(np.std(fold_scores, ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def feature_ablation(rf_model: Pipeline, x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame, y_test: np.ndarray) -> pd.DataFrame:
    base_auc = roc_auc_score(y_test, rf_model.predict_proba(x_test)[:, 1])
    importances = pd.Series(rf_model.named_steps["model"].feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

    rows = [{"ablation": "none", "n_removed": 0, "test_auc": float(base_auc)}]
    for k in [1, 2, 3, 5, 8]:
        removed = importances.index[:k].tolist()
        keep = [c for c in FEATURE_COLS if c not in removed]

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=350, random_state=SEED, class_weight="balanced")),
        ])
        model.fit(x_train[keep], y_train)
        auc = roc_auc_score(y_test, model.predict_proba(x_test[keep])[:, 1])
        rows.append(
            {
                "ablation": f"remove_top_{k}",
                "n_removed": k,
                "removed_features": ",".join(removed),
                "test_auc": float(auc),
                "auc_drop_vs_base": float(base_auc - auc),
            }
        )
    return pd.DataFrame(rows)


def biological_validation(df: pd.DataFrame, top_features: list[str]) -> pd.DataFrame:
    rows = []
    for feat in top_features:
        sub = df[[feat, "compound", "concentration"]].dropna()

        compound_groups = [g[feat].values for _, g in sub.groupby("compound") if len(g) >= 3]
        if len(compound_groups) >= 2:
            h_stat, p_kw = kruskal(*compound_groups)
        else:
            h_stat, p_kw = np.nan, np.nan

        if sub["concentration"].nunique() > 2:
            rho, p_spear = spearmanr(sub[feat], sub["concentration"], nan_policy="omit")
        else:
            rho, p_spear = np.nan, np.nan

        rows.append(
            {
                "feature": feat,
                "kruskal_h": float(h_stat) if not np.isnan(h_stat) else np.nan,
                "kruskal_p": float(p_kw) if not np.isnan(p_kw) else np.nan,
                "spearman_rho_vs_concentration": float(rho) if not np.isnan(rho) else np.nan,
                "spearman_p": float(p_spear) if not np.isnan(p_spear) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def train_cnn_and_resnet(
    x_img: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
) -> tuple[TorchResult, TorchResult, float, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tiny CNN (1-channel)
    train_ds = GrayDataset(np.expand_dims(x_img[idx_train], 1), y[idx_train])
    val_ds = GrayDataset(np.expand_dims(x_img[idx_val], 1), y[idx_val])
    test_ds = GrayDataset(np.expand_dims(x_img[idx_test], 1), y[idx_test])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    cnn = TinyCNN().to(device)
    tracemalloc.start()
    t0 = time.perf_counter()
    train_torch_binary(cnn, train_loader, val_loader, CNN_EPOCHS, LEARNING_RATE, device)
    cnn_time = time.perf_counter() - t0
    cnn_peak = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()

    val_proba = predict_proba_torch(cnn, val_loader, device)
    cnn_thr = select_threshold(y[idx_val], val_proba)
    cnn_test_proba = predict_proba_torch(cnn, test_loader, device)

    cnn_result = TorchResult(
        proba_test=cnn_test_proba,
        train_time_sec=float(cnn_time),
        peak_mem_mb=float(cnn_peak),
        used_pretrained=False,
    )

    # ResNet-18 transfer baseline (3-channel duplicate)
    x3 = np.repeat(np.expand_dims(x_img, 1), 3, axis=1)
    train_ds_r = GrayDataset(x3[idx_train], y[idx_train])
    val_ds_r = GrayDataset(x3[idx_val], y[idx_val])
    test_ds_r = GrayDataset(x3[idx_test], y[idx_test])

    train_loader_r = DataLoader(train_ds_r, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_r = DataLoader(val_ds_r, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_r = DataLoader(test_ds_r, batch_size=BATCH_SIZE, shuffle=False)

    resnet = ResNet18Binary().to(device)
    tracemalloc.start()
    t0 = time.perf_counter()
    train_torch_binary(resnet, train_loader_r, val_loader_r, RESNET_EPOCHS, LEARNING_RATE, device)
    res_time = time.perf_counter() - t0
    res_peak = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()

    val_proba_r = predict_proba_torch(resnet, val_loader_r, device)
    res_thr = select_threshold(y[idx_val], val_proba_r)
    res_test_proba = predict_proba_torch(resnet, test_loader_r, device)

    # Return thresholds separately for confusion matrix
    return (
        TorchResult(res_test_proba, float(cnn_time), float(cnn_peak), False),
        TorchResult(res_test_proba, float(res_time), float(res_peak), bool(resnet.used_pretrained)),
        float(cnn_thr),
        float(res_thr),
    )


def save_calibration_plot(y_true: np.ndarray, probs: dict[str, np.ndarray], out_path: Path) -> pd.DataFrame:
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    rows = []
    for name, p in probs.items():
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy="uniform")
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
        rows.append({"model": name, "ece": ece_score(y_true, p, n_bins=10)})
    plt.title("Calibration Curves")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return pd.DataFrame(rows)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def run(root: Path, target_total: int = 1600) -> None:
    set_seed(SEED)

    final_figures = root / "final_figures"
    final_tables = root / "final_tables"
    results_summary = root / "results_summary"
    for d in [final_figures, final_tables, results_summary]:
        d.mkdir(parents=True, exist_ok=True)

    images_dir, csv_path = ensure_data(root)
    meta = load_metadata(csv_path)
    meta["batch_name"] = meta.apply(infer_batch_name, axis=1)

    rows = collect_rows(meta, images_dir, target_total=target_total)
    if len(rows) < 1500:
        raise RuntimeError(f"Collected {len(rows)} rows; expected at least 1500.")

    df, x_img = build_dataset(rows, images_dir)
    y = (df["group"] == "Group B").astype(int).values

    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(idx, test_size=0.20, random_state=SEED, stratify=y)
    idx_train, idx_val = train_test_split(idx_train, test_size=0.15, random_state=SEED, stratify=y[idx_train])

    x_train = df.iloc[idx_train][FEATURE_COLS]
    y_train = y[idx_train]
    x_test = df.iloc[idx_test][FEATURE_COLS]
    y_test = y[idx_test]

    lr, rf, train_times, peak_mems = fit_classical_models(x_train, y_train)

    lr_proba = lr.predict_proba(x_test)[:, 1]
    rf_proba = rf.predict_proba(x_test)[:, 1]

    cnn_res, resnet_res, cnn_thr, res_thr = train_cnn_and_resnet(x_img, y, idx_train, idx_val, idx_test)

    model_probs = {
        "logistic_regression": lr_proba,
        "random_forest": rf_proba,
        "cnn": cnn_res.proba_test,
        "resnet18_transfer": resnet_res.proba_test,
    }

    metrics_rows = []
    for name, prob in model_probs.items():
        row = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, (prob >= 0.5).astype(int))),
            "roc_auc": float(roc_auc_score(y_test, prob)),
        }
        row.update(per_class_auc(y_test, prob))
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)

    # DeLong tests against feature model baseline (RF)
    delong_cnn_vs_rf = delong_test(y_test, cnn_res.proba_test, rf_proba)
    delong_resnet_vs_rf = delong_test(y_test, resnet_res.proba_test, rf_proba)

    # Confusion matrices
    save_confusion_matrix(y_test, (rf_proba >= 0.5).astype(int), final_figures / "figure10_confusion_matrix_rf.png", "Confusion Matrix - Random Forest")
    save_confusion_matrix(y_test, (cnn_res.proba_test >= cnn_thr).astype(int), final_figures / "figure11_confusion_matrix_cnn.png", "Confusion Matrix - CNN")
    save_confusion_matrix(y_test, (resnet_res.proba_test >= res_thr).astype(int), final_figures / "figure12_confusion_matrix_resnet18.png", "Confusion Matrix - ResNet-18")

    # Calibration + ECE
    ece_df = save_calibration_plot(y_test, model_probs, final_figures / "figure13_calibration_curves.png")

    # PCA batch effects before/after
    save_pca_batch_effects(df, FEATURE_COLS, final_figures / "figure14_pca_batch_before.png", final_figures / "figure15_pca_batch_after.png")

    # Feature ablation
    ablation_df = feature_ablation(rf, x_train, y_train, x_test, y_test)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=ablation_df, x="ablation", y="test_auc")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.title("Feature Ablation Study (Random Forest)")
    plt.tight_layout()
    plt.savefig(final_figures / "figure16_feature_ablation.png", dpi=180)
    plt.close()

    # Biological validation on top features
    top_features = pd.Series(rf.named_steps["model"].feature_importances_, index=FEATURE_COLS).sort_values(ascending=False).head(6).index.tolist()
    bio_df = biological_validation(df, top_features)

    # Nested CV on training set
    nested_df = nested_cv_assessment(x_train, y_train)

    # Computational cost table
    cost_df = pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                "train_time_sec": train_times["logistic_regression"],
                "peak_memory_mb": peak_mems["logistic_regression"],
            },
            {
                "model": "random_forest",
                "train_time_sec": train_times["random_forest"],
                "peak_memory_mb": peak_mems["random_forest"],
            },
            {
                "model": "cnn",
                "train_time_sec": cnn_res.train_time_sec,
                "peak_memory_mb": cnn_res.peak_mem_mb,
            },
            {
                "model": "resnet18_transfer",
                "train_time_sec": resnet_res.train_time_sec,
                "peak_memory_mb": resnet_res.peak_mem_mb,
                "used_pretrained_weights": resnet_res.used_pretrained,
            },
        ]
    )

    # Save tables
    df.to_csv(final_tables / "full_dataset_features_1500plus.csv", index=False)
    metrics_df.to_csv(results_summary / "advanced_model_metrics.csv", index=False)
    ece_df.to_csv(results_summary / "advanced_calibration_ece.csv", index=False)
    ablation_df.to_csv(results_summary / "advanced_feature_ablation.csv", index=False)
    bio_df.to_csv(results_summary / "advanced_biological_validation.csv", index=False)
    nested_df.to_csv(results_summary / "advanced_nested_cv.csv", index=False)
    cost_df.to_csv(results_summary / "advanced_computational_cost.csv", index=False)

    with open(results_summary / "advanced_delong_tests.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "cnn_vs_random_forest": delong_cnn_vs_rf,
                "resnet18_vs_random_forest": delong_resnet_vs_rf,
            },
            f,
            indent=2,
        )

    summary = {
        "n_samples": int(len(df)),
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "metrics_file": "results_summary/advanced_model_metrics.csv",
        "delong_file": "results_summary/advanced_delong_tests.json",
    }
    with open(results_summary / "advanced_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Completed full advanced pipeline")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    run(project_root, target_total=1600)
