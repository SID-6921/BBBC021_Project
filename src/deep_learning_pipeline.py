from __future__ import annotations

from pathlib import Path
import json
import random
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import RocCurveDisplay, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data_loader import load_metadata, load_multichannel_image
from preprocess import clean_image, normalize_image, resize_image
from robustness_pipeline import ZIP_URLS, ensure_data

SEED = 42
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-3


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_batch_name(row: pd.Series) -> str:
    for col in row.index:
        if "pathname" not in col.lower() or pd.isna(row[col]):
            continue
        match = re.search(r"Week\d+_\d+", str(row[col]))
        if match:
            return match.group(0)
    table_number = row.get("TableNumber")
    return f"Table_{table_number}"


def build_binary_group(compound_series: pd.Series) -> pd.Series:
    top_compound = compound_series.value_counts().idxmax()
    return np.where(compound_series == top_compound, "Group A", "Group B")


def collect_rows(meta: pd.DataFrame, images_dir: Path, target_per_batch: int = 80) -> list[tuple[int, pd.Series]]:
    requested_batches = [Path(url).stem.replace("BBBC021_v1_images_", "") for url in ZIP_URLS]
    channel_cols = [c for c in meta.columns if "filename" in c.lower()]

    rows: list[tuple[int, pd.Series]] = []
    for batch_name in requested_batches:
        batch_df = meta[meta["batch_name"] == batch_name]
        count = 0
        for idx, row in batch_df.iterrows():
            try:
                _img, _chs = load_multichannel_image(row, images_dir, channel_cols)
                rows.append((idx, row))
                count += 1
            except Exception:
                continue
            if count >= target_per_batch:
                break
    return rows


class ImageDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = torch.from_numpy(images.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32)).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
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

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def preprocess_row(row: pd.Series, images_dir: Path) -> np.ndarray:
    fused, _ = load_multichannel_image(row, images_dir)
    img = clean_image(normalize_image(resize_image(fused, (IMG_SIZE, IMG_SIZE))))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": []}

    for _epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(xb)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += float(loss.item()) * len(xb)

        history["train_loss"].append(train_loss / len(train_loader.dataset))
        history["val_loss"].append(val_loss / len(val_loader.dataset))

    return history


def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            proba = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            preds.extend(proba.tolist())
    return np.array(preds, dtype=np.float32)


def select_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t = 0.5
    best_acc = -1.0
    for t in thresholds:
        pred = (y_proba >= t).astype(np.int64)
        acc = float(accuracy_score(y_true, pred))
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t


def run(root: Path, target_per_batch: int = 80) -> None:
    seed_everything()

    final_figures = root / "final_figures"
    final_tables = root / "final_tables"
    results_summary = root / "results_summary"
    for d in [final_figures, final_tables, results_summary]:
        d.mkdir(parents=True, exist_ok=True)

    images_dir, csv_path = ensure_data(root)
    meta = load_metadata(csv_path)
    meta["batch_name"] = meta.apply(infer_batch_name, axis=1)

    rows = collect_rows(meta, images_dir, target_per_batch=target_per_batch)
    if not rows:
        raise RuntimeError("No rows available for deep learning pipeline.")

    compounds = pd.Series([str(row.get("Image_Metadata_Compound", "unknown")) for _, row in rows])
    groups = build_binary_group(compounds)
    y = np.array([1 if g == "Group B" else 0 for g in groups], dtype=np.int64)

    x_list = [preprocess_row(row, images_dir) for _, row in rows]
    x = np.stack(x_list, axis=0)

    idx_all = np.arange(len(x))
    idx_train, idx_test = train_test_split(idx_all, test_size=0.30, random_state=SEED, stratify=y)
    idx_train, idx_val = train_test_split(idx_train, test_size=0.20, random_state=SEED, stratify=y[idx_train])

    train_ds = ImageDataset(x[idx_train], y[idx_train])
    val_ds = ImageDataset(x[idx_val], y[idx_val])
    test_ds = ImageDataset(x[idx_test], y[idx_test])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCNN().to(device)

    history = train_model(model, train_loader, val_loader, device)

    y_val = y[idx_val]
    y_test = y[idx_test]
    y_val_proba = predict_proba(model, val_loader, device)
    best_threshold = select_threshold(y_val, y_val_proba)
    y_proba = predict_proba(model, test_loader, device)
    y_pred = (y_proba >= best_threshold).astype(np.int64)

    metrics = {
        "cnn_accuracy": float(accuracy_score(y_test, y_pred)),
        "cnn_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "n_images": int(len(x)),
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "device": str(device),
        "decision_threshold": float(best_threshold),
    }

    with open(results_summary / "deep_learning_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame(
        {
            "sample_index": idx_test,
            "y_true": y_test,
            "y_proba_group_b": y_proba,
            "y_pred": y_pred,
        }
    )
    pred_df.to_csv(final_tables / "deep_learning_predictions.csv", index=False)

    plt.figure(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, name="CNN")
    plt.title("Figure 9: CNN ROC Curve")
    plt.tight_layout()
    plt.savefig(final_figures / "figure9_deep_learning_roc.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.8))
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.title("Figure 9: CNN Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(final_figures / "figure9_deep_learning_training.png", dpi=180)
    plt.close()

    comparison_rows = []
    class_path = results_summary / "classification_metrics.json"
    if class_path.exists():
        base = json.loads(class_path.read_text(encoding="utf-8"))
        comparison_rows.append(
            {
                "model": "logistic_regression",
                "accuracy": base["logistic_regression"]["accuracy"],
                "roc_auc": base["logistic_regression"]["roc_auc"],
            }
        )
        comparison_rows.append(
            {
                "model": "random_forest",
                "accuracy": base["random_forest"]["accuracy"],
                "roc_auc": base["random_forest"]["roc_auc"],
            }
        )

    comparison_rows.append(
        {
            "model": "cnn",
            "accuracy": metrics["cnn_accuracy"],
            "roc_auc": metrics["cnn_roc_auc"],
        }
    )

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(results_summary / "model_comparison_final.csv", index=False)

    print("Completed deep learning pipeline")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    run(project_root, target_per_batch=80)
