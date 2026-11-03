"""Estimate per-box visibility from GT geometry when the dataset has none.

FastTracker's gt.txt stores a constant 1.0 in the visibility column. For oracle
detections / oracle VOS we need occlusion, so we approximate it: on a coarse grid,
boxes whose bottom edge is lower in the image are assumed closer to the camera
(true for elevated CCTV views) and hide the boxes behind them.
"""
from __future__ import annotations

import numpy as np

from rad.io.mot import F, V, X, Y, W, H, group_by_frame


def _frame_visibility(boxes: np.ndarray, width: int, height: int, cell: int = 4) -> np.ndarray:
    gw, gh = int(np.ceil(width / cell)) + 1, int(np.ceil(height / cell)) + 1
    owner = -np.ones((gh, gw), dtype=int)
    order = np.argsort(boxes[:, 3])  # far (small bottom y) first, near last
    cells = []
    for k in range(len(boxes)):
        x1, y1, x2, y2 = boxes[k]
        c1, r1 = int(np.clip(x1 // cell, 0, gw)), int(np.clip(y1 // cell, 0, gh))
        c2, r2 = int(np.clip(np.ceil(x2 / cell), 0, gw)), int(np.clip(np.ceil(y2 / cell), 0, gh))
        cells.append((r1, r2, c1, c2))
    for k in order:
        r1, r2, c1, c2 = cells[k]
        if r2 > r1 and c2 > c1:
            owner[r1:r2, c1:c2] = k
    vis = np.ones(len(boxes))
    for k, (r1, r2, c1, c2) in enumerate(cells):
        n = (r2 - r1) * (c2 - c1)
        if n > 0:
            vis[k] = (owner[r1:r2, c1:c2] == k).sum() / n
    return vis


def estimate_visibility(gt: np.ndarray, width: int | None = None, height: int | None = None,
                        cell: int = 4) -> np.ndarray:
    """Return a copy of gt with column V replaced by the geometric visibility estimate."""
    out = gt.copy()
    if len(gt) == 0:
        return out
    width = int(width or np.ceil((gt[:, X] + gt[:, W]).max()) + 1)
    height = int(height or np.ceil((gt[:, Y] + gt[:, H]).max()) + 1)
    order = np.argsort(gt[:, F], kind="stable")
    vis_sorted = []
    for f, rows in group_by_frame(gt).items():
        boxes = np.stack([rows[:, X], rows[:, Y], rows[:, X] + rows[:, W], rows[:, Y] + rows[:, H]], 1)
        vis_sorted.append(_frame_visibility(boxes, width, height, cell))
    out[order, V] = np.concatenate(vis_sorted)
    return out


def needs_visibility(gt: np.ndarray) -> bool:
    return len(gt) > 0 and bool(np.allclose(gt[:, V], gt[0, V]))
