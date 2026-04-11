from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import random
import re
import time
import urllib.request
import zipfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
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
from sklearn.preprocessing import StandardScaler, label_binarize
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from data_loader import load_metadata, load_multichannel_image
from detect import detect_spots
from features import compute_features
from preprocess import clean_image, normalize_image, resize_image

SEED = 42
BATCH_SIZE = 16          # reduced from 32: saves ~50% GPU/CPU RAM during DL training
LEARNING_RATE = 1e-3
RESNET_FINETUNE_LR = 1e-4
CNN_EPOCHS = 6
RESNET_HEAD_EPOCHS = 10
RESNET_FINETUNE_EPOCHS = 6

DETECT_CONFIGS = {
    "sensitive": {"min_area": 8, "max_area": 1800, "min_mean_intensity": 35.0, "adaptive_block_size": 31, "adaptive_c": -3},
    "default": {"min_area": 10, "max_area": 1800, "min_mean_intensity": 45.0, "adaptive_block_size": 35, "adaptive_c": -4},
    "conservative": {"min_area": 14, "max_area": 1600, "min_mean_intensity": 55.0, "adaptive_block_size": 39, "adaptive_c": -5},
}

FULL_ZIP_URLS = [
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22123.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22141.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22161.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22361.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22381.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22401.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week2_24121.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week2_24141.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week2_24161.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week2_24361.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week2_24381.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week2_24401.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week3_25421.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week3_25441.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week3_25461.zip",
    "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week3_25681.zip",
]
CSV_URL = "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_image.csv"
MOA_URL = "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_moa.csv"

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


@dataclass
class ModelResult:
    name: str
    proba_test: np.ndarray
    pred_test: np.ndarray
    train_time_sec: float
    peak_memory_mb: float
    used_pretrained: bool = False


class GrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class TinyCNNMulti(nn.Module):
    def __init__(self, n_classes: int):
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
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ResNet18Multi(nn.Module):
    def __init__(self, n_classes: int, pretrained: bool):
        super().__init__()
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            base = models.resnet18(weights=weights)
        else:
            base = models.resnet18(weights=None)

        base.fc = nn.Linear(base.fc.in_features, n_classes)
        self.model = base
        self.used_pretrained = pretrained

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_data(root: Path) -> tuple[Path, Path, Path]:
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
    moa_path = meta_dir / "BBBC021_v1_moa.csv"
    if not csv_path.exists():
        urllib.request.urlretrieve(CSV_URL, csv_path)
    if not moa_path.exists():
        urllib.request.urlretrieve(MOA_URL, moa_path)

    return images_dir, csv_path, moa_path


def infer_batch_name(row: pd.Series) -> str:
    for col in row.index:
        if "pathname" not in col.lower() or pd.isna(row[col]):
            continue
        match = re.search(r"Week\d+_\d+", str(row[col]))
        if match:
            return match.group(0)
    return f"Table_{row.get('TableNumber', 'NA')}"


def attach_moa_labels(meta: pd.DataFrame, moa: pd.DataFrame, min_rows: int = 1500) -> pd.DataFrame:
    out = meta.copy()
    out["compound_norm"] = out["Image_Metadata_Compound"].astype(str).str.strip().str.lower()
    out["conc_round"] = pd.to_numeric(out["Image_Metadata_Concentration"], errors="coerce").round(6)

    moa2 = moa.copy()
    moa2["compound_norm"] = moa2["compound"].astype(str).str.strip().str.lower()
    moa2["conc_round"] = pd.to_numeric(moa2["concentration"], errors="coerce").round(6)

    merged = out.merge(
        moa2[["compound_norm", "conc_round", "moa"]],
        how="left",
        on=["compound_norm", "conc_round"],
    )

    moa_labeled = merged.dropna(subset=["moa"]).copy()
    top12_moa = moa_labeled["moa"].value_counts().index[:12].tolist()
    moa_labeled = moa_labeled[moa_labeled["moa"].isin(top12_moa)].copy()

    if len(moa_labeled) >= min_rows:
        moa_labeled["class_name"] = moa_labeled["moa"].astype(str)
        moa_labeled["label_source"] = "moa"
        return moa_labeled

    # Fallback: use 12 most frequent compound-treatment classes from available metadata subset.
    comp = out.copy()
    top12_compounds = comp["Image_Metadata_Compound"].astype(str).value_counts().index[:12].tolist()
    comp = comp[comp["Image_Metadata_Compound"].astype(str).isin(top12_compounds)].copy()
    comp["class_name"] = comp["Image_Metadata_Compound"].astype(str)
    comp["label_source"] = "compound"
    return comp


def collect_rows(meta: pd.DataFrame, images_dir: Path, target_total: int = 1600) -> list[tuple[int, pd.Series]]:
    channel_cols = [c for c in meta.columns if "filename" in c.lower()]
    available: list[tuple[int, pd.Series]] = []

    for idx, row in meta.iterrows():
        try:
            _img, _chs = load_multichannel_image(row, images_dir, channel_cols)
            available.append((idx, row))
        except Exception:
            continue

    if not available:
        return []

    cls_counts: dict[str, int] = {}
    for _, r in available:
        c = str(r.get("class_name", "unknown"))
        cls_counts[c] = cls_counts.get(c, 0) + 1

    top_classes = [k for k, _ in sorted(cls_counts.items(), key=lambda kv: kv[1], reverse=True)[:12]]
    filtered = [(i, r) for i, r in available if str(r.get("class_name", "unknown")) in top_classes]

    per_class_target = int(math.ceil(target_total / max(1, len(top_classes))))
    by_class: dict[str, list[tuple[int, pd.Series]]] = {c: [] for c in top_classes}
    for item in filtered:
        c = str(item[1].get("class_name", "unknown"))
        by_class[c].append(item)

    rows: list[tuple[int, pd.Series]] = []
    for c in top_classes:
        rows.extend(by_class[c][:per_class_target])

    return rows[:target_total]


def extract_feature_record(image_id: str, row: pd.Series, fused: np.ndarray, config: dict) -> dict:
    img_detect = clean_image(normalize_image(resize_image(fused, (256, 256))))
    det = detect_spots(img_detect, **config)
    feat = compute_features(
        image_gray=img_detect,
        mask=det["mask"],
        spot_count=det["spot_count"],
        image_id=image_id,
        group=str(row.get("class_name", "unknown")),
        spots=det.get("spots", []),
    )
    feat["class_name"] = str(row.get("class_name", "unknown"))
    feat["compound"] = str(row.get("Image_Metadata_Compound", "unknown"))
    feat["concentration"] = float(row.get("Image_Metadata_Concentration", np.nan))
    feat["batch_name"] = infer_batch_name(row)
    return feat


def build_dataset(rows: list[tuple[int, pd.Series]], images_dir: Path, config: dict, dl_size: int) -> tuple[pd.DataFrame, np.ndarray]:
    records: list[dict] = []
    images: list[np.ndarray] = []

    for idx, row in rows:
        image_id = f"image_{idx:05d}"
        fused, _ = load_multichannel_image(row, images_dir)

        records.append(extract_feature_record(image_id, row, fused, config))

        img_dl = clean_image(normalize_image(resize_image(fused, (dl_size, dl_size))))
        images.append(img_dl.astype(np.float32) / 255.0)

    return pd.DataFrame(records), np.stack(images, axis=0)


def split_indices(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx_all = np.arange(len(y))
    idx_train, idx_test = train_test_split(idx_all, test_size=0.20, random_state=SEED, stratify=y)
    idx_train, idx_val = train_test_split(idx_train, test_size=0.15, random_state=SEED, stratify=y[idx_train])
    return idx_train, idx_val, idx_test


def get_peak_rss_mb(proc: psutil.Process) -> float:
    return float(proc.memory_info().rss / (1024 * 1024))


def train_multiclass_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    optimizer_params: list[torch.nn.Parameter] | None = None,
) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    params = optimizer_params if optimizer_params is not None else list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    proc = psutil.Process()
    peak = get_peak_rss_mb(proc)
    t0 = time.perf_counter()

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            peak = max(peak, get_peak_rss_mb(proc))

        model.eval()
        with torch.no_grad():
            for xb, _ in val_loader:
                _ = model(xb.to(device))
                peak = max(peak, get_peak_rss_mb(proc))

    elapsed = time.perf_counter() - t0
    return float(elapsed), float(peak)


def predict_proba_multiclass(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            probs = torch.softmax(model(xb.to(device)), dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.vstack(all_probs)


def run_resnet_training(
    pretrained: bool,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
) -> tuple[ModelResult, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ResNet expects 3-channel 224x224 images
    x_train3 = np.repeat(np.expand_dims(x_train, 1), 3, axis=1)
    x_val3 = np.repeat(np.expand_dims(x_val, 1), 3, axis=1)
    x_test3 = np.repeat(np.expand_dims(x_test, 1), 3, axis=1)

    train_ds = GrayDataset(x_train3, y_train)
    val_ds = GrayDataset(x_val3, y_val)
    test_ds = GrayDataset(x_test3, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ResNet18Multi(n_classes=n_classes, pretrained=pretrained).to(device)

    if pretrained:
        # Stage 1: freeze everything except FC
        for p in model.model.parameters():
            p.requires_grad = False
        for p in model.model.fc.parameters():
            p.requires_grad = True
        t1, peak1 = train_multiclass_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=RESNET_HEAD_EPOCHS,
            lr=LEARNING_RATE,
            optimizer_params=list(model.model.fc.parameters()),
        )

        # Stage 2: unfreeze last two residual blocks + fc and fine-tune
        for name, p in model.model.named_parameters():
            if name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc"):
                p.requires_grad = True
            else:
                p.requires_grad = False

        trainable = [p for p in model.parameters() if p.requires_grad]
        t2, peak2 = train_multiclass_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=RESNET_FINETUNE_EPOCHS,
            lr=RESNET_FINETUNE_LR,
            optimizer_params=trainable,
        )
        elapsed = t1 + t2
        peak = max(peak1, peak2)
    else:
        elapsed, peak = train_multiclass_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=RESNET_HEAD_EPOCHS,
            lr=LEARNING_RATE,
        )

    proba = predict_proba_multiclass(model, test_loader, device)
    pred = np.argmax(proba, axis=1)
    return ModelResult(
        name="resnet18_pretrained" if pretrained else "resnet18_scratch",
        proba_test=proba,
        pred_test=pred,
        train_time_sec=elapsed,
        peak_memory_mb=peak,
        used_pretrained=pretrained,
    ), float(accuracy_score(y_test, pred))


def run_cnn_training(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
) -> ModelResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = GrayDataset(np.expand_dims(x_train, 1), y_train)
    val_ds = GrayDataset(np.expand_dims(x_val, 1), y_val)
    test_ds = GrayDataset(np.expand_dims(x_test, 1), y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = TinyCNNMulti(n_classes=n_classes).to(device)
    elapsed, peak = train_multiclass_model(model, train_loader, val_loader, device, epochs=CNN_EPOCHS, lr=LEARNING_RATE)

    proba = predict_proba_multiclass(model, test_loader, device)
    pred = np.argmax(proba, axis=1)
    return ModelResult(
        name="cnn_scratch",
        proba_test=proba,
        pred_test=pred,
        train_time_sec=elapsed,
        peak_memory_mb=peak,
        used_pretrained=False,
    )


def fit_classical_models(x_train: pd.DataFrame, y_train: np.ndarray) -> tuple[Pipeline, Pipeline, dict[str, float], dict[str, float]]:
    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2500, random_state=SEED)),
    ])
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=450, random_state=SEED, class_weight="balanced_subsample")),
    ])

    proc = psutil.Process()
    timings: dict[str, float] = {}
    peaks: dict[str, float] = {}

    t0 = time.perf_counter()
    lr.fit(x_train, y_train)
    timings["logistic_regression"] = time.perf_counter() - t0
    peaks["logistic_regression"] = get_peak_rss_mb(proc)

    t0 = time.perf_counter()
    rf.fit(x_train, y_train)
    timings["random_forest"] = time.perf_counter() - t0
    peaks["random_forest"] = get_peak_rss_mb(proc)

    return lr, rf, timings, peaks


def multiclass_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    classes = np.arange(y_prob.shape[1])
    y_bin = label_binarize(y_true, classes=classes)
    return float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"))


def per_class_auc(y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str]) -> dict[str, float]:
    classes = np.arange(len(class_names))
    y_bin = label_binarize(y_true, classes=classes)
    aucs = roc_auc_score(y_bin, y_prob, average=None, multi_class="ovr")
    return {f"auc_{class_names[i]}": float(aucs[i]) for i in range(len(class_names))}


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    conf = np.max(y_prob, axis=1)
    pred = np.argmax(y_prob, axis=1)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        ece += (np.sum(mask) / n) * abs(np.mean(correct[mask]) - np.mean(conf[mask]))
    return float(ece)


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = 200) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    accs = []
    aucs = []
    eces = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ys = y_true[idx]
        ps = y_prob[idx]
        accs.append(accuracy_score(ys, np.argmax(ps, axis=1)))
        try:
            aucs.append(multiclass_auc(ys, ps))
        except Exception:
            continue
        eces.append(ece_score(ys, ps))

    def q(v: list[float]) -> tuple[float, float]:
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    a_l, a_u = q(accs)
    u_l, u_u = q(aucs)
    e_l, e_u = q(eces)
    return {
        "acc_ci_low": a_l,
        "acc_ci_high": a_u,
        "auc_ci_low": u_l,
        "auc_ci_high": u_u,
        "ece_ci_low": e_l,
        "ece_ci_high": e_u,
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


def delong_test_multiclass_micro(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict[str, float]:
    # Micro-style flattening of one-vs-rest targets/probabilities for a binary DeLong application.
    classes = np.arange(p1.shape[1])
    y_bin = label_binarize(y_true, classes=classes).ravel()
    pred1 = p1.ravel()
    pred2 = p2.ravel()

    order = np.argsort(-y_bin)
    label_1_count = int(np.sum(y_bin))
    preds = np.vstack([pred1, pred2])[:, order]
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


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 9))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, xticks_rotation=45, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_calibration_plot(y_true: np.ndarray, probs: dict[str, np.ndarray], out_path: Path) -> pd.DataFrame:
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    rows = []
    for name, p in probs.items():
        conf = np.max(p, axis=1)
        corr = (np.argmax(p, axis=1) == y_true).astype(int)
        frac_pos, mean_pred = calibration_curve(corr, conf, n_bins=12, strategy="uniform")
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
        rows.append({"model": name, "ece": ece_score(y_true, p)})
    plt.title("Calibration Curves (Confidence Reliability)")
    plt.xlabel("Predicted confidence")
    plt.ylabel("Observed accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return pd.DataFrame(rows)


def save_pca_batch_effects(df: pd.DataFrame, out_before: Path, out_after: Path) -> None:
    x_before = StandardScaler().fit_transform(df[FEATURE_COLS])
    p_before = PCA(n_components=2, random_state=SEED).fit_transform(x_before)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=p_before[:, 0], y=p_before[:, 1], hue=df["batch_name"], s=25)
    plt.title("PCA Batch Effects Before Normalization")
    plt.tight_layout()
    plt.savefig(out_before, dpi=180)
    plt.close()

    df_norm = df.copy()
    df_norm[FEATURE_COLS] = df_norm[FEATURE_COLS].astype(float)
    for b, bdf in df_norm.groupby("batch_name"):
        idx = bdf.index
        mu = bdf[FEATURE_COLS].mean()
        sd = bdf[FEATURE_COLS].std(ddof=0).replace(0, 1.0)
        df_norm.loc[idx, FEATURE_COLS] = (bdf[FEATURE_COLS] - mu) / sd

    x_after = StandardScaler().fit_transform(df_norm[FEATURE_COLS])
    p_after = PCA(n_components=2, random_state=SEED).fit_transform(x_after)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=p_after[:, 0], y=p_after[:, 1], hue=df_norm["batch_name"], s=25)
    plt.title("PCA Batch Effects After Normalization")
    plt.tight_layout()
    plt.savefig(out_after, dpi=180)
    plt.close()


def nested_cv_assessment(x_train: pd.DataFrame, y_train: np.ndarray) -> pd.DataFrame:
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    lr_grid = {"model__C": [0.2, 1.0, 4.0]}
    rf_grid = {"model__n_estimators": [300, 450], "model__max_depth": [None, 10, 18]}

    lr_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2500, random_state=SEED)),
    ])
    rf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(random_state=SEED, class_weight="balanced_subsample")),
    ])

    rows = []
    for name, pipe, grid in [("logistic_regression", lr_pipe, lr_grid), ("random_forest", rf_pipe, rf_grid)]:
        scores = []
        for tr_idx, te_idx in outer_cv.split(x_train, y_train):
            inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
            gs = GridSearchCV(pipe, grid, scoring="roc_auc_ovr_weighted", cv=inner, n_jobs=2)
            gs.fit(x_train.iloc[tr_idx], y_train[tr_idx])
            proba = gs.predict_proba(x_train.iloc[te_idx])
            scores.append(multiclass_auc(y_train[te_idx], proba))
        rows.append({"model": name, "nested_cv_auc_mean": float(np.mean(scores)), "nested_cv_auc_std": float(np.std(scores, ddof=0))})
    return pd.DataFrame(rows)


def feature_ablation(rf_model: Pipeline, x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame, y_test: np.ndarray) -> pd.DataFrame:
    base = multiclass_auc(y_test, rf_model.predict_proba(x_test))
    imp = pd.Series(rf_model.named_steps["model"].feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

    rows = [{"ablation": "none", "n_removed": 0, "test_auc": float(base), "auc_drop_vs_base": 0.0, "removed_features": ""}]
    for k in [1, 2, 3, 5, 8]:
        rem = imp.index[:k].tolist()
        keep = [c for c in FEATURE_COLS if c not in rem]
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=450, random_state=SEED, class_weight="balanced_subsample")),
        ])
        model.fit(x_train[keep], y_train)
        auc = multiclass_auc(y_test, model.predict_proba(x_test[keep]))
        rows.append({"ablation": f"remove_top_{k}", "n_removed": k, "test_auc": float(auc), "auc_drop_vs_base": float(base - auc), "removed_features": ",".join(rem)})
    return pd.DataFrame(rows)


def biological_validation(df: pd.DataFrame, top_features: list[str]) -> pd.DataFrame:
    rows = []
    for feat in top_features:
        sub = df[[feat, "compound", "concentration", "class_name"]].dropna()
        groups = [g[feat].values for _, g in sub.groupby("class_name") if len(g) >= 3]
        if len(groups) >= 2:
            h, p_kw = kruskal(*groups)
        else:
            h, p_kw = np.nan, np.nan

        if sub["concentration"].nunique() > 2:
            rho, p_s = spearmanr(sub[feat], sub["concentration"], nan_policy="omit")
        else:
            rho, p_s = np.nan, np.nan

        rows.append({
            "feature": feat,
            "kruskal_h": float(h) if not np.isnan(h) else np.nan,
            "kruskal_p": float(p_kw) if not np.isnan(p_kw) else np.nan,
            "spearman_rho_vs_concentration": float(rho) if not np.isnan(rho) else np.nan,
            "spearman_p": float(p_s) if not np.isnan(p_s) else np.nan,
        })
    return pd.DataFrame(rows)


def save_feature_correlation(df: pd.DataFrame, out_path: Path) -> None:
    corr = df[FEATURE_COLS].corr(method="spearman")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
    plt.title("Supplementary Figure S1. Spearman Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def evaluate_models_with_ci(y_test: np.ndarray, class_names: list[str], model_probs: dict[str, np.ndarray], model_preds: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, probs in model_probs.items():
        pred = model_preds[name]
        row = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "roc_auc_macro_ovr": multiclass_auc(y_test, probs),
            "ece": ece_score(y_test, probs),
        }
        row.update(per_class_auc(y_test, probs, class_names))
        row.update(bootstrap_ci(y_test, probs))
        rows.append(row)
    return pd.DataFrame(rows)


def run_robustness_same_split(
    rows: list[tuple[int, pd.Series]],
    images_dir: Path,
    split_df: pd.DataFrame,
    class_to_idx: dict[str, int],
    out_path: Path,
) -> pd.DataFrame:
    split_map = dict(zip(split_df["image_id"], split_df["split"]))
    robust_rows = []

    for cfg_name, cfg in DETECT_CONFIGS.items():
        feat_records = []
        for idx, row in rows:
            image_id = f"image_{idx:05d}"
            fused, _ = load_multichannel_image(row, images_dir)
            rec = extract_feature_record(image_id, row, fused, cfg)
            rec["split"] = split_map.get(image_id, "")
            rec["label_idx"] = class_to_idx.get(rec["class_name"], -1)
            feat_records.append(rec)

        rdf = pd.DataFrame(feat_records)
        rdf = rdf[(rdf["split"].isin(["train", "test"])) & (rdf["label_idx"] >= 0)].copy()

        train_df = rdf[rdf["split"] == "train"]
        test_df = rdf[rdf["split"] == "test"]
        x_tr = train_df[FEATURE_COLS]
        y_tr = train_df["label_idx"].astype(int).values
        x_te = test_df[FEATURE_COLS]
        y_te = test_df["label_idx"].astype(int).values

        lr = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2500, random_state=SEED)),
        ])
        rf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=350, random_state=SEED, class_weight="balanced_subsample")),
        ])
        lr.fit(x_tr, y_tr)
        rf.fit(x_tr, y_tr)

        lr_prob = lr.predict_proba(x_te)
        rf_prob = rf.predict_proba(x_te)

        robust_rows.append({
            "config_name": cfg_name,
            "n_test": int(len(y_te)),
            "logistic_accuracy": float(accuracy_score(y_te, np.argmax(lr_prob, axis=1))),
            "logistic_roc_auc_macro_ovr": multiclass_auc(y_te, lr_prob),
            "rf_accuracy": float(accuracy_score(y_te, np.argmax(rf_prob, axis=1))),
            "rf_roc_auc_macro_ovr": multiclass_auc(y_te, rf_prob),
        })

    out_df = pd.DataFrame(robust_rows)
    out_df.to_csv(out_path, index=False)
    return out_df


def write_revision_notes(results_summary: Path) -> None:
    text = """
External validation note: We observed a gap between nested cross-validation performance and held-out test performance. In this revision cycle, priority was given to implementing reviewer-requested methodological controls (pretrained transfer learning, multiclass expansion, statistical testing, calibration, and split-consistent robustness). Independent external dataset validation (e.g., BBBC014/BBBC020/RxRx1) remains a planned follow-up experiment and is explicitly acknowledged as a limitation.
""".strip()
    (results_summary / "external_validation_acknowledgement.txt").write_text(text, encoding="utf-8")


def run(root: Path, target_total: int = 1600) -> None:
    set_seed(SEED)

    final_figures = root / "final_figures"
    final_tables = root / "final_tables"
    results_summary = root / "results_summary"
    for d in [final_figures, final_tables, results_summary]:
        d.mkdir(parents=True, exist_ok=True)

    images_dir, csv_path, moa_path = ensure_data(root)
    meta = load_metadata(csv_path)
    moa = pd.read_csv(moa_path)
    meta = attach_moa_labels(meta, moa)

    rows = collect_rows(meta, images_dir, target_total=target_total)
    if len(rows) < 1500:
        print(f"Warning: collected {len(rows)} rows for 12-class setup in downloaded subset; proceeding with available data.")

    df_default, x_img = build_dataset(rows, images_dir, DETECT_CONFIGS["default"], dl_size=224)

    class_names = sorted(df_default["class_name"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    y = df_default["class_name"].map(class_to_idx).astype(int).values

    idx_train, idx_val, idx_test = split_indices(y)
    split_df = pd.DataFrame({"image_id": df_default["image_id"], "split": "unused"})
    split_df.loc[idx_train, "split"] = "train"
    split_df.loc[idx_val, "split"] = "val"
    split_df.loc[idx_test, "split"] = "test"
    split_df.to_csv(results_summary / "main_split_indices.csv", index=False)

    x_train = df_default.iloc[idx_train][FEATURE_COLS]
    y_train = y[idx_train]
    x_test = df_default.iloc[idx_test][FEATURE_COLS]
    y_test = y[idx_test]

    lr, rf, t_cls, m_cls = fit_classical_models(x_train, y_train)
    lr_prob = lr.predict_proba(x_test)
    rf_prob = rf.predict_proba(x_test)

    # Deep models
    cnn_res = run_cnn_training(x_img[idx_train], y[idx_train], x_img[idx_val], y[idx_val], x_img[idx_test], y[idx_test], n_classes=len(class_names))
    res_scratch, _ = run_resnet_training(False, x_img[idx_train], y[idx_train], x_img[idx_val], y[idx_val], x_img[idx_test], y[idx_test], n_classes=len(class_names))
    res_pre, _ = run_resnet_training(True, x_img[idx_train], y[idx_train], x_img[idx_val], y[idx_val], x_img[idx_test], y[idx_test], n_classes=len(class_names))

    model_probs = {
        "logistic_regression": lr_prob,
        "random_forest": rf_prob,
        "cnn_scratch": cnn_res.proba_test,
        "resnet18_scratch": res_scratch.proba_test,
        "resnet18_pretrained": res_pre.proba_test,
    }
    model_preds = {
        "logistic_regression": np.argmax(lr_prob, axis=1),
        "random_forest": np.argmax(rf_prob, axis=1),
        "cnn_scratch": cnn_res.pred_test,
        "resnet18_scratch": res_scratch.pred_test,
        "resnet18_pretrained": res_pre.pred_test,
    }

    metrics_df = evaluate_models_with_ci(y_test, class_names, model_probs, model_preds)

    # DeLong tests
    delong_cnn_vs_rf = delong_test_multiclass_micro(y_test, model_probs["cnn_scratch"], model_probs["random_forest"])
    delong_resnet_pre_vs_rf = delong_test_multiclass_micro(y_test, model_probs["resnet18_pretrained"], model_probs["random_forest"])

    # Figures and tables
    save_confusion_matrix(y_test, model_preds["logistic_regression"], class_names, final_figures / "figure11_confusion_matrix_lr_12class.png", "12-class Confusion Matrix - Logistic Regression")
    save_confusion_matrix(y_test, model_preds["random_forest"], class_names, final_figures / "figure12_confusion_matrix_rf_12class.png", "12-class Confusion Matrix - Random Forest")
    save_confusion_matrix(y_test, model_preds["resnet18_pretrained"], class_names, final_figures / "figure13_confusion_matrix_resnet_pretrained_12class.png", "12-class Confusion Matrix - ResNet-18 Pretrained")

    ece_df = save_calibration_plot(y_test, model_probs, final_figures / "figure14_calibration_curves_multiclass.png")
    save_pca_batch_effects(df_default, final_figures / "figure15_pca_batch_before.png", final_figures / "figure16_pca_batch_after.png")

    ablation_df = feature_ablation(rf, x_train, y_train, x_test, y_test)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=ablation_df, x="ablation", y="test_auc")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.title("Feature Ablation Study (Random Forest, 12-class)")
    plt.tight_layout()
    plt.savefig(final_figures / "figure17_feature_ablation.png", dpi=180)
    plt.close()

    save_feature_correlation(df_default, final_figures / "supplementary_figure_s1_feature_correlation.png")

    top_features = pd.Series(rf.named_steps["model"].feature_importances_, index=FEATURE_COLS).sort_values(ascending=False).head(6).index.tolist()
    bio_df = biological_validation(df_default, top_features)
    nested_df = nested_cv_assessment(x_train, y_train)

    cost_df = pd.DataFrame([
        {"model": "logistic_regression", "train_time_sec": t_cls["logistic_regression"], "peak_memory_mb": m_cls["logistic_regression"], "used_pretrained_weights": "NA"},
        {"model": "random_forest", "train_time_sec": t_cls["random_forest"], "peak_memory_mb": m_cls["random_forest"], "used_pretrained_weights": "NA"},
        {"model": "cnn_scratch", "train_time_sec": cnn_res.train_time_sec, "peak_memory_mb": cnn_res.peak_memory_mb, "used_pretrained_weights": False},
        {"model": "resnet18_scratch", "train_time_sec": res_scratch.train_time_sec, "peak_memory_mb": res_scratch.peak_memory_mb, "used_pretrained_weights": False},
        {"model": "resnet18_pretrained", "train_time_sec": res_pre.train_time_sec, "peak_memory_mb": res_pre.peak_memory_mb, "used_pretrained_weights": True},
    ])

    robust_df = run_robustness_same_split(rows, images_dir, split_df, class_to_idx, results_summary / "robustness_classification.csv")

    # Save outputs
    df_default.to_csv(final_tables / "full_dataset_features_1500plus.csv", index=False)
    metrics_df.to_csv(results_summary / "advanced_model_metrics.csv", index=False)
    ece_df.to_csv(results_summary / "advanced_calibration_ece.csv", index=False)
    ablation_df.to_csv(results_summary / "advanced_feature_ablation.csv", index=False)
    bio_df.to_csv(results_summary / "advanced_biological_validation.csv", index=False)
    nested_df.to_csv(results_summary / "advanced_nested_cv.csv", index=False)
    cost_df.to_csv(results_summary / "advanced_computational_cost.csv", index=False)

    model_comparison = metrics_df[metrics_df["model"].isin(["cnn_scratch", "resnet18_scratch", "resnet18_pretrained"])][["model", "accuracy", "roc_auc_macro_ovr", "ece"]]
    model_comparison.to_csv(results_summary / "model_comparison_transfer_learning.csv", index=False)

    with open(results_summary / "advanced_delong_tests.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "cnn_scratch_vs_random_forest": delong_cnn_vs_rf,
                "resnet18_pretrained_vs_random_forest": delong_resnet_pre_vs_rf,
            },
            f,
            indent=2,
        )

    write_revision_notes(results_summary)

    summary = {
        "n_samples": int(len(df_default)),
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "n_classes": int(len(class_names)),
        "metrics_file": "results_summary/advanced_model_metrics.csv",
        "robustness_file": "results_summary/robustness_classification.csv",
        "delong_file": "results_summary/advanced_delong_tests.json",
    }
    with open(results_summary / "advanced_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Completed full-scale multiclass revision pipeline")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    run(project_root, target_total=1600)
