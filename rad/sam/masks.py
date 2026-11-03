"""Mask representations used by the SMP windows.

BoxMask: an axis-aligned rectangle (oracle backend, cheap).
ArrayMask: a boolean HxW array (SAM 3 output).
Both expose area(), bbox() and inter_area(boxes) for IoMA = |M ∩ B| / |M|.
"""
from __future__ import annotations

import numpy as np

from rad.boxes import intersection_matrix


class BoxMask:
    def __init__(self, xyxy):
        self.xyxy = np.asarray(xyxy, dtype=float).reshape(4)

    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)

    def bbox(self) -> np.ndarray:
        return self.xyxy.copy()

    def inter_area(self, boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros(0)
        return intersection_matrix(self.xyxy[None], boxes)[0]

    def is_empty(self) -> bool:
        return self.area() <= 0


class ArrayMask:
    def __init__(self, mask: np.ndarray):
        self.mask = np.asarray(mask, dtype=bool)
        self._area = int(self.mask.sum())
        if self._area:
            ys, xs = np.nonzero(self.mask)
            self._bbox = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float)
            # integral image for fast rectangle sums
            self._ii = np.pad(self.mask.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        else:
            self._bbox = np.zeros(4)
            self._ii = None

    def area(self) -> float:
        return float(self._area)

    def bbox(self) -> np.ndarray:
        return self._bbox.copy()

    def inter_area(self, boxes: np.ndarray) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        if self._ii is None or len(boxes) == 0:
            return np.zeros(len(boxes))
        h, w = self.mask.shape
        x1 = np.clip(np.floor(boxes[:, 0]), 0, w).astype(int)
        y1 = np.clip(np.floor(boxes[:, 1]), 0, h).astype(int)
        x2 = np.clip(np.ceil(boxes[:, 2]), 0, w).astype(int)
        y2 = np.clip(np.ceil(boxes[:, 3]), 0, h).astype(int)
        ii = self._ii
        s = ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1]
        return np.where((x2 > x1) & (y2 > y1), s, 0).astype(float)

    def is_empty(self) -> bool:
        return self._area == 0


def ioma(mask, boxes: np.ndarray) -> np.ndarray:
    """Intersection over Mask Area of a mask with each box (xyxy)."""
    a = mask.area()
    if a <= 0:
        return np.zeros(len(boxes))
    return mask.inter_area(boxes) / a
