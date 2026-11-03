"""CLEAR-MOT style GT matching that also returns event-level identity switches.

The per-frame matching keeps previous GT<->track correspondences when IoU still
passes the threshold (CLEAR MOT rule), then solves the rest with Hungarian. An
identity switch is counted when a GT object is matched to a different track ID
than the one it was last matched to.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from rad.boxes import iou_matrix
from rad.io.mot import ID, X, Y, W, H, group_by_frame


@dataclass
class ClearResult:
    matches: pd.DataFrame      # frame, gt_id, trk_id, iou
    switches: pd.DataFrame     # frame, gt_id, prev_trk, new_trk
    num_gt: int
    num_fp: int
    num_fn: int
    num_switches: int
    num_frag: int

    @property
    def mota(self) -> float:
        return 1.0 - (self.num_fn + self.num_fp + self.num_switches) / max(self.num_gt, 1)

    def summary(self) -> dict:
        return dict(num_gt=self.num_gt, fp=self.num_fp, fn=self.num_fn,
                    idsw=self.num_switches, frag=self.num_frag, mota=self.mota)


def _xyxy(rows: np.ndarray) -> np.ndarray:
    return np.stack([rows[:, X], rows[:, Y], rows[:, X] + rows[:, W], rows[:, Y] + rows[:, H]], 1)


def clear_match(gt: np.ndarray, trk: np.ndarray, iou_thresh: float = 0.5,
                frames: tuple[int, int] | None = None) -> ClearResult:
    """Match GT and tracker outputs frame by frame (optionally restricted to [f0, f1])."""
    g_by = group_by_frame(gt)
    t_by = group_by_frame(trk)
    all_frames = sorted(set(g_by) | set(t_by))
    if frames is not None:
        all_frames = [f for f in all_frames if frames[0] <= f <= frames[1]]
    last: dict[int, int] = {}          # gt_id -> last matched trk_id
    tracked_prev: set[int] = set()     # gt ids matched in the previous frame
    m_rows, s_rows = [], []
    n_gt = n_fp = n_fn = n_sw = n_frag = 0
    for f in all_frames:
        g = g_by.get(f, np.zeros((0, 9)))
        t = t_by.get(f, np.zeros((0, 9)))
        n_gt += len(g)
        gid = g[:, ID].astype(int)
        tid = t[:, ID].astype(int)
        ious = iou_matrix(_xyxy(g), _xyxy(t)) if len(g) and len(t) else np.zeros((len(g), len(t)))
        pairs = []
        used_g, used_t = set(), set()
        # keep previous correspondences
        t_index = {v: k for k, v in enumerate(tid)}
        for gi, g_id in enumerate(gid):
            prev = last.get(g_id)
            if prev is not None and prev in t_index:
                ti = t_index[prev]
                if ti not in used_t and ious[gi, ti] >= iou_thresh:
                    pairs.append((gi, ti))
                    used_g.add(gi)
                    used_t.add(ti)
        rg = [i for i in range(len(g)) if i not in used_g]
        rt = [j for j in range(len(t)) if j not in used_t]
        if rg and rt:
            sub = ious[np.ix_(rg, rt)]
            cost = np.where(sub >= iou_thresh, 1.0 - sub, 1e6)
            r, c = linear_sum_assignment(cost)
            for a, b in zip(r, c):
                if cost[a, b] < 1e6:
                    pairs.append((rg[a], rt[b]))
        matched_now = set()
        for gi, ti in pairs:
            g_id, t_id = int(gid[gi]), int(tid[ti])
            prev = last.get(g_id)
            if prev is not None and prev != t_id:
                n_sw += 1
                s_rows.append(dict(frame=f, gt_id=g_id, prev_trk=prev, new_trk=t_id))
            if prev is not None and g_id not in tracked_prev:
                n_frag += 1
            last[g_id] = t_id
            matched_now.add(g_id)
            m_rows.append(dict(frame=f, gt_id=g_id, trk_id=t_id, iou=float(ious[gi, ti])))
        n_fn += len(g) - len(pairs)
        n_fp += len(t) - len(pairs)
        tracked_prev = matched_now
    return ClearResult(
        matches=pd.DataFrame(m_rows, columns=["frame", "gt_id", "trk_id", "iou"]),
        switches=pd.DataFrame(s_rows, columns=["frame", "gt_id", "prev_trk", "new_trk"]),
        num_gt=n_gt, num_fp=n_fp, num_fn=n_fn, num_switches=n_sw, num_frag=n_frag,
    )


def idf1(gt: np.ndarray, trk: np.ndarray, iou_thresh: float = 0.5) -> dict:
    """Global ID F1 (Ristani et al. 2016) via bipartite matching of whole trajectories."""
    g_by, t_by = group_by_frame(gt), group_by_frame(trk)
    g_ids = np.unique(gt[:, ID]).astype(int) if len(gt) else np.array([], int)
    t_ids = np.unique(trk[:, ID]).astype(int) if len(trk) else np.array([], int)
    gi = {v: k for k, v in enumerate(g_ids)}
    ti = {v: k for k, v in enumerate(t_ids)}
    overlap = np.zeros((len(g_ids), len(t_ids)))
    for f in set(g_by) & set(t_by):
        g, t = g_by[f], t_by[f]
        ious = iou_matrix(_xyxy(g), _xyxy(t))
        a, b = np.where(ious >= iou_thresh)
        for p, q in zip(a, b):
            overlap[gi[int(g[p, ID])], ti[int(t[q, ID])]] += 1
    g_len = np.array([np.sum(gt[:, ID] == v) for v in g_ids])
    t_len = np.array([np.sum(trk[:, ID] == v) for v in t_ids])
    n_g, n_t = len(g_ids), len(t_ids)
    if n_g == 0 or n_t == 0:
        return dict(idf1=0.0, idp=0.0, idr=0.0, idtp=0)
    # cost matrix of size (n_g + n_t) square, standard IDF1 formulation
    big = np.zeros((n_g + n_t, n_g + n_t))
    fp = t_len[None, :] - overlap      # tracker frames not matched to this gt
    fn = g_len[:, None] - overlap
    big[:n_g, :n_t] = fp + fn
    big[:n_g, n_t:] = 1e9
    big[n_g:, :n_t] = 1e9
    for i in range(n_g):
        big[i, n_t + i] = g_len[i]
    for j in range(n_t):
        big[n_g + j, j] = t_len[j]
    r, c = linear_sum_assignment(big)
    idtp = sum(overlap[a, b] for a, b in zip(r, c) if a < n_g and b < n_t)
    idp = idtp / max(t_len.sum(), 1)
    idr = idtp / max(g_len.sum(), 1)
    return dict(idf1=2 * idtp / max(t_len.sum() + g_len.sum(), 1), idp=idp, idr=idr, idtp=int(idtp))


def evaluate(gt: np.ndarray, trk: np.ndarray, iou_thresh: float = 0.5) -> dict:
    res = clear_match(gt, trk, iou_thresh)
    out = res.summary()
    out.update(idf1(gt, trk, iou_thresh))
    return out
