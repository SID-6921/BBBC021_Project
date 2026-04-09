from __future__ import annotations

import cv2
import numpy as np


def detect_spots(
    image_gray: np.ndarray,
    min_area: int = 8,
    max_area: int = 2000,
    min_mean_intensity: float = 30.0,
    adaptive_block_size: int = 35,
    adaptive_c: int = -5,
) -> dict:
    """Detect bright spots using adaptive thresholding and contour filtering."""
    if adaptive_block_size % 2 == 0:
        adaptive_block_size += 1

    smoothed = cv2.medianBlur(image_gray, 3)
    binary = cv2.adaptiveThreshold(
        smoothed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        adaptive_block_size,
        adaptive_c,
    )

    open_kernel = np.ones((3, 3), np.uint8)
    close_kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = []
    mask = np.zeros_like(image_gray)
    spots: list[dict] = []

    for c in contours:
        area = cv2.contourArea(c)

        if area < min_area or area > max_area:
            continue

        temp_mask = np.zeros_like(image_gray)
        cv2.drawContours(temp_mask, [c], -1, 255, thickness=cv2.FILLED)
        mean_intensity = float(cv2.mean(image_gray, mask=temp_mask)[0])
        if mean_intensity < min_mean_intensity:
            continue

        x, y, w, h = cv2.boundingRect(c)
        m = cv2.moments(c)
        center_x = int(m["m10"] / m["m00"]) if m["m00"] > 0 else x + w // 2
        center_y = int(m["m01"] / m["m00"]) if m["m00"] > 0 else y + h // 2

        kept.append(c)
        cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
        spots.append(
            {
                "area": float(area),
                "mean_intensity": mean_intensity,
                "bbox": (x, y, w, h),
                "center": (center_x, center_y),
            }
        )

    return {
        "mask": mask,
        "binary": cleaned,
        "contours": kept,
        "spots": spots,
        "spot_count": len(kept),
    }


def create_overlay(image_gray: np.ndarray, contours: list[np.ndarray], spots: list[dict] | None = None) -> np.ndarray:
    """Draw presentation-ready contour overlays with detection labels."""
    overlay = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(overlay, contours, -1, (40, 220, 80), 1)

    if spots is None:
        spots = []

    for spot in spots:
        x, y, w, h = spot["bbox"]
        cx, cy = spot["center"]
        radius = max(2, int(min(w, h) / 2))
        cv2.circle(overlay, (cx, cy), radius, (0, 255, 255), 1)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 180, 0), 1)

    label = f"Detections: {len(contours)}"
    cv2.rectangle(overlay, (8, 8), (220, 36), (20, 20, 20), -1)
    cv2.putText(overlay, label, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay
