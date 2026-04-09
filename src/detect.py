from __future__ import annotations

import cv2
import numpy as np


def detect_spots(
    image_gray: np.ndarray,
    min_area: int = 8,
    max_area: int = 2000,
) -> dict:
    """Detect bright spots using Otsu thresholding + contour filtering."""
    blurred = cv2.GaussianBlur(image_gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = []
    mask = np.zeros_like(image_gray)

    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            kept.append(c)
            cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)

    return {
        "mask": mask,
        "binary": opened,
        "contours": kept,
        "spot_count": len(kept),
    }


def create_overlay(image_gray: np.ndarray, contours: list[np.ndarray]) -> np.ndarray:
    """Draw detected contours over a grayscale image."""
    overlay = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)

    for c in contours:
        m = cv2.moments(c)
        if m["m00"] > 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            cv2.circle(overlay, (cx, cy), 1, (0, 0, 255), -1)

    return overlay
