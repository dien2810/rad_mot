"""Ignore regions (FastTracker class 9 'ignore_region').

An ignore region is an annotated area, not an object. Treating it as a "don't care"
zone, the same rule is applied to every box source:
  - GT:        rows of the ignore classes are removed; GT objects lying mostly inside
               a region of the same frame are removed too;
  - detections and tracker outputs: boxes lying mostly inside a region are dropped,
               so they are neither tracked nor counted as false positives.
"Mostly inside" = intersection / box area > max_ioa (default 0.5).
Regions are matched frame by frame (each gt.txt row of an ignore class is a region
in that frame).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from rad.boxes import area, intersection_matrix
from rad.io.mot import C, F, H, W, X, Y


def _xyxy(rows: np.ndarray) -> np.ndarray:
    return np.stack([rows[:, X], rows[:, Y], rows[:, X] + rows[:, W], rows[:, Y] + rows[:, H]], 1)


def split_ignore(gt: np.ndarray, ignore_classes: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    """(object rows, region rows) of a raw gt array."""
    ign = [int(c) for c in ignore_classes or ()]
    if len(gt) == 0 or not ign:
        return gt, np.zeros((0, gt.shape[1] if gt.ndim == 2 else 9))
    m = np.isin(gt[:, C].astype(int), ign)
    return gt[~m], gt[m]


def inside_regions(rows: np.ndarray, regions: np.ndarray, max_ioa: float = 0.5) -> np.ndarray:
    """Boolean mask: row lies mostly inside an ignore region of its own frame."""
    out = np.zeros(len(rows), bool)
    if len(rows) == 0 or len(regions) == 0:
        return out
    reg_by_frame: dict[int, np.ndarray] = {}
    for f in np.unique(regions[:, F].astype(int)):
        reg_by_frame[int(f)] = _xyxy(regions[regions[:, F].astype(int) == f])
    frames = rows[:, F].astype(int)
    for f, reg in reg_by_frame.items():
        idx = np.where(frames == f)[0]
        if len(idx) == 0:
            continue
        boxes = _xyxy(rows[idx])
        inter = intersection_matrix(boxes, reg).sum(1)  # regions rarely overlap; sum is an upper bound
        out[idx] = inter / np.maximum(area(boxes), 1e-9) > max_ioa
    return out


def drop_in_regions(rows: np.ndarray, regions: np.ndarray, max_ioa: float = 0.5) -> tuple[np.ndarray, int]:
    """Rows without those lying mostly inside an ignore region, and how many were dropped."""
    m = inside_regions(rows, regions, max_ioa)
    return rows[~m], int(m.sum())


def clean_gt(gt: np.ndarray, ignore_classes: Iterable[int], max_ioa: float = 0.5
             ) -> tuple[np.ndarray, np.ndarray, int]:
    """(GT objects outside ignore regions, region rows, number of objects removed for being inside)."""
    objs, regions = split_ignore(gt, ignore_classes)
    objs, n_in = drop_in_regions(objs, regions, max_ioa)
    return objs, regions, n_in
