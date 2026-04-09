from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd


def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    """Load BBBC021 metadata CSV."""
    return pd.read_csv(csv_path)


def get_channel_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that likely contain image file names."""
    return [c for c in df.columns if "filename" in c.lower()]


def assign_group_labels(df: pd.DataFrame, preferred_col: str | None = None) -> pd.DataFrame:
    """Create a simple Group A vs Group B label from metadata.

    Strategy:
    - use preferred_col if provided
    - otherwise, use first non-file metadata text column with >1 unique value
    - largest category -> Group A, all others -> Group B
    """
    out = df.copy()

    candidate_col = None
    if preferred_col and preferred_col in out.columns:
        candidate_col = preferred_col
    else:
        ignored = set(get_channel_columns(out))
        for c in out.columns:
            if c in ignored:
                continue
            if out[c].dtype == object and out[c].nunique(dropna=True) > 1:
                candidate_col = c
                break

    if candidate_col is None:
        out["group_raw"] = np.where(np.arange(len(out)) % 2 == 0, "even_index", "odd_index")
    else:
        out["group_raw"] = out[candidate_col].astype(str)

    top_value = out["group_raw"].value_counts().idxmax()
    out["group"] = np.where(out["group_raw"] == top_value, "Group A", "Group B")
    return out


def load_multichannel_image(
    row: pd.Series,
    images_dir: str | Path,
    channel_columns: Iterable[str] | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Load available channels and return a fused grayscale image plus channels."""
    images_dir = Path(images_dir)
    cols = list(channel_columns) if channel_columns is not None else [c for c in row.index if "filename" in c.lower()]

    def _build_candidates(file_col: str, file_value: str) -> list[Path]:
        file_name = Path(file_value).name
        candidates: list[Path] = []

        # Direct filename path from metadata value.
        candidates.append(images_dir / file_value)

        # Try matching paired pathname column, e.g. Image_PathName_DAPI for Image_FileName_DAPI.
        path_col = file_col.replace("FileName", "PathName")
        if path_col in row and not pd.isna(row[path_col]):
            raw_path = str(row[path_col]).replace("\\", "/")
            parts = [p for p in raw_path.split("/") if p]
            # Keep only tail parts from the first BBBC021 marker onward.
            start_idx = 0
            for i, p in enumerate(parts):
                if p.lower() == "bbbc021":
                    start_idx = i + 1
                    break
            rel_parts = parts[start_idx:]
            if rel_parts:
                candidates.append(images_dir.joinpath(*rel_parts, file_name))

        # Final fallback: search by basename recursively.
        candidates.append(images_dir / file_name)
        return candidates

    channels: list[np.ndarray] = []
    for col in cols:
        if col not in row or pd.isna(row[col]):
            continue
        file_value = str(row[col])
        chosen_path: Path | None = None

        for candidate in _build_candidates(col, file_value):
            if candidate.exists():
                chosen_path = candidate
                break

        if chosen_path is None:
            matches = list(images_dir.rglob(Path(file_value).name))
            if matches:
                chosen_path = matches[0]

        if chosen_path is None:
            continue

        img = cv2.imread(str(chosen_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        channels.append(img)

    if not channels:
        raise FileNotFoundError("No channel images were found for the provided metadata row.")

    stacked = np.stack(channels, axis=0).astype(np.float32)
    fused = np.mean(stacked, axis=0)
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    return fused, channels
