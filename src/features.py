from __future__ import annotations

import numpy as np


def compute_features(
    image_gray: np.ndarray,
    mask: np.ndarray,
    spot_count: int,
    image_id: str,
    group: str,
    spots: list[dict] | None = None,
) -> dict:
    if spots is None:
        spots = []

    total_intensity = float(np.sum(image_gray))
    mean_intensity = float(np.mean(image_gray))
    intensity_variance = float(np.var(image_gray))
    area_covered_px = int(np.count_nonzero(mask))
    area_covered_ratio = float(area_covered_px / image_gray.size)
    density_spots_per_px = float(spot_count / image_gray.size)
    density_spots_per_10k_px = float(density_spots_per_px * 10000.0)

    areas = np.array([float(s["area"]) for s in spots], dtype=np.float32)
    if areas.size == 0:
        spot_area_mean = 0.0
        spot_area_std = 0.0
        spot_area_median = 0.0
        spot_area_q25 = 0.0
        spot_area_q75 = 0.0
        small_spot_fraction = 0.0
        medium_spot_fraction = 0.0
        large_spot_fraction = 0.0
    else:
        spot_area_mean = float(np.mean(areas))
        spot_area_std = float(np.std(areas))
        spot_area_median = float(np.median(areas))
        spot_area_q25 = float(np.percentile(areas, 25))
        spot_area_q75 = float(np.percentile(areas, 75))

        small = int(np.sum(areas < 25))
        medium = int(np.sum((areas >= 25) & (areas < 100)))
        large = int(np.sum(areas >= 100))
        total = float(areas.size)
        small_spot_fraction = float(small / total)
        medium_spot_fraction = float(medium / total)
        large_spot_fraction = float(large / total)

    return {
        "image_id": image_id,
        "group": group,
        "spot_count": int(spot_count),
        "mean_intensity": mean_intensity,
        "total_intensity": total_intensity,
        "intensity_variance": intensity_variance,
        "area_covered_px": area_covered_px,
        "area_covered_ratio": area_covered_ratio,
        "density_spots_per_px": density_spots_per_px,
        "density_spots_per_10k_px": density_spots_per_10k_px,
        "spot_area_mean": spot_area_mean,
        "spot_area_std": spot_area_std,
        "spot_area_median": spot_area_median,
        "spot_area_q25": spot_area_q25,
        "spot_area_q75": spot_area_q75,
        "small_spot_fraction": small_spot_fraction,
        "medium_spot_fraction": medium_spot_fraction,
        "large_spot_fraction": large_spot_fraction,
    }
