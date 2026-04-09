from __future__ import annotations

import numpy as np


def compute_features(
    image_gray: np.ndarray,
    mask: np.ndarray,
    spot_count: int,
    image_id: str,
    group: str,
) -> dict:
    total_intensity = float(np.sum(image_gray))
    avg_brightness = float(np.mean(image_gray))
    area_covered_px = int(np.count_nonzero(mask))
    area_covered_ratio = float(area_covered_px / image_gray.size)

    return {
        "image_id": image_id,
        "group": group,
        "spot_count": int(spot_count),
        "avg_brightness": avg_brightness,
        "total_intensity": total_intensity,
        "area_covered_px": area_covered_px,
        "area_covered_ratio": area_covered_ratio,
    }
