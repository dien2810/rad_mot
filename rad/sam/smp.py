"""Re-implementation of Selective Mask Propagation (Holmberg, arXiv 2606.13033).

No official code or full hyper-parameters were released, so this follows the paper's
description and exposes every threshold in SMPParams. Report it as a re-implementation.

Pipeline
  1. Triggers (frame, track_id) come from a dispatch policy. The SMP baseline policy
     fires when the assignment margin < tau_entry or a lost track is re-found after
     >= gap_G frames. Witness windows are opened for tracks overlapping the target.
  2. Seed: walk back from the trigger to the latest run of n_seed consecutive "clean"
     frames (margin >= tau_seed and no overlapping track box).
  3. Propagate a VOS mask from the seed box; per frame, find the track the mask lies in
     via IoMA = |M ∩ B| / |M|.
  4. Exit when, for n_exit consecutive frames after the trigger, the mask sits in a
     single track, that track's margin has recovered, and it is spatially isolated.
  5. Outcomes: CLEAN (same track) / SWAP (different track -> relabel from the swap
     frame) / STALE (left the seed track before the trigger) / DEGRADED (mask
     collapsed) / LOST (mask empty > max_occluded frames) / EDGE (left the image) /
     END (no exit within max_len) / SEED_FAIL.
     Only SWAP modifies the base tracker output.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd

from rad.boxes import iou_matrix
from rad.io.mot import F, ID, X, Y, W, H, group_by_frame
from rad.sam.masks import ioma


@dataclass
class SMPParams:
    tau_entry: float = 0.05
    gap_G: int = 5
    witness: bool = True
    witness_iou: float = 0.0       # open witness windows for tracks with IoU > this
    n_seed: int = 3
    tau_seed: float = 0.2
    iso_iou: float = 0.0           # "no nearby overlapping box" = IoU <= iso_iou
    max_lookback: int = 60
    n_exit: int = 5
    tau_ioma: float = 0.5
    tau_box_cov: float = 0.3       # also require |M ∩ B| / |B| >= this (mask fills the box)
    tau_ioma_second: float = 0.2
    mask_iso_frac: float = 0.15
    max_len: int = 150
    degrade_ratio: float = 0.25    # non-empty mask smaller than this x seed area = collapsed
    degrade_frames: int = 5
    max_occluded: int = 30         # empty mask (object hidden) tolerated this many frames
    edge_px: float = 4.0
    image_size: tuple = (1920, 1080)  # (W, H) for the EDGE test


@dataclass
class Window:
    track_id: int
    entry_frame: int
    trigger: str
    witness_of: Optional[int] = None
    seed_frame: Optional[int] = None
    end_frame: Optional[int] = None
    outcome: str = "SEED_FAIL"
    settled_id: Optional[int] = None
    swap_frame: Optional[int] = None
    frames: int = 0
    applied: bool = False  # set by apply_swaps: this window actually relabelled tracks


class _Index:
    """Fast per-frame / per-track lookups over a tracker output."""

    def __init__(self, tracks: np.ndarray, log: pd.DataFrame):
        self.by_frame = {}
        for f, rows in group_by_frame(tracks).items():
            ids = rows[:, ID].astype(int)
            boxes = np.stack([rows[:, X], rows[:, Y], rows[:, X] + rows[:, W], rows[:, Y] + rows[:, H]], 1)
            self.by_frame[f] = (ids, boxes)
        self.track_box: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
        for f, (ids, boxes) in self.by_frame.items():
            for i, b in zip(ids, boxes):
                self.track_box[int(i)][f] = b
        self.margin: dict[tuple[int, int], float] = {}
        if len(log):
            for f, t, m in zip(log["frame"].to_numpy(), log["track_id"].to_numpy(),
                               log["margin_det"].to_numpy()):
                key = (int(f), int(t))
                self.margin[key] = min(float(m), self.margin.get(key, np.inf))

    def frame(self, f: int):
        return self.by_frame.get(f, (np.zeros(0, int), np.zeros((0, 4))))

    def isolated(self, f: int, tid: int, box: np.ndarray, iso_iou: float) -> bool:
        ids, boxes = self.frame(f)
        others = boxes[ids != tid]
        if len(others) == 0:
            return True
        return bool(iou_matrix(box[None], others)[0].max() <= iso_iou)


def smp_triggers(log: pd.DataFrame, p: SMPParams) -> pd.DataFrame:
    """The SMP baseline dispatch rule: low margin or re-found after a gap."""
    if len(log) == 0:
        return pd.DataFrame(columns=["frame", "track_id", "trigger"])
    m = log["margin_det"] < p.tau_entry
    g = log["gap"] >= p.gap_G
    out = log.loc[m | g, ["frame", "track_id"]].copy()
    out["trigger"] = np.where(g[m | g], "gap", "margin")
    return out.sort_values(["frame", "track_id"]).drop_duplicates(["frame", "track_id"]).reset_index(drop=True)


def _seed(idx: _Index, tid: int, t: int, p: SMPParams) -> Optional[int]:
    run, cand = 0, None
    boxes = idx.track_box.get(tid, {})
    for f in range(t - 1, max(t - p.max_lookback, 1) - 1, -1):
        box = boxes.get(f)
        m = idx.margin.get((f, tid))
        clean = box is not None and m is not None and m >= p.tau_seed and idx.isolated(f, tid, box, p.iso_iou)
        if clean:
            run += 1
            if run == 1:
                cand = f
            if run >= p.n_seed:
                return cand
        else:
            run = 0
    return None


def _run_window(idx: _Index, backend, w: Window, seq_len: int, p: SMPParams) -> Window:
    tid, t = w.track_id, w.entry_frame
    seed = _seed(idx, tid, t, p)
    if seed is None:
        w.outcome = "SEED_FAIL"
        return w
    w.seed_frame = seed
    seed_box = idx.track_box[tid][seed]
    seed_area = None
    assigned: list[tuple[int, Optional[int]]] = []
    io_hist: list[tuple[int, dict]] = []  # frame -> {track_id: IoMA} (for the swap frame)
    exit_run = degrade_run = occl_run = 0
    outcome, last = "END", seed
    img_w, img_h = p.image_size
    max_len = min(p.max_len, seq_len - seed + 1)
    for f, mask, _score in backend.propagate(seed, seed_box, max_len):
        last = f
        w.frames += 1
        a = mask.area()
        if seed_area is None:
            seed_area = max(a, 1.0)
        if a <= 0:  # object hidden: SAM keeps memory and may recover after the occlusion
            occl_run += 1
            if occl_run > p.max_occluded:
                outcome = "LOST"
                break
            assigned.append((f, None))
            exit_run = 0
            continue
        occl_run = 0
        degrade_run = degrade_run + 1 if a < p.degrade_ratio * seed_area else 0
        if degrade_run >= p.degrade_frames:
            outcome = "DEGRADED"
            break
        ids, boxes = idx.frame(f)
        io = ioma(mask, boxes) if len(boxes) else np.zeros(0)
        order = np.argsort(-io)
        io_hist.append((f, {int(ids[k]): float(io[k]) for k in order[:4] if io[k] > 0}))
        best = None
        if len(io) and io[order[0]] >= p.tau_ioma:
            bb0 = boxes[order[0]]
            box_area = max((bb0[2] - bb0[0]) * (bb0[3] - bb0[1]), 1e-6)
            if mask.inter_area(bb0[None])[0] / box_area >= p.tau_box_cov:
                best = int(ids[order[0]])
        second = float(io[order[1]]) if len(io) > 1 else 0.0
        bb = mask.bbox()
        touches = bb[0] <= p.edge_px or bb[1] <= p.edge_px or bb[2] >= img_w - p.edge_px or bb[3] >= img_h - p.edge_px
        if touches and best is None:
            outcome = "EDGE"
            break
        assigned.append((f, best))
        if f < t:
            if best is not None and best != tid:
                outcome = "STALE"
                break
            continue
        ok = False
        if best is not None and second < p.tau_ioma_second:
            m = idx.margin.get((f, best))
            if m is not None and m >= p.tau_seed:
                bbox_best = boxes[order[0]]
                others = boxes[ids != best]
                iso = True
                if len(others):
                    iso = iou_matrix(bbox_best[None], others)[0].max() <= p.iso_iou
                    iso = iso and mask.inter_area(others).sum() / a <= p.mask_iso_frac
                ok = bool(iso)
        exit_run = exit_run + 1 if ok else 0
        if exit_run >= p.n_exit:
            w.settled_id = best
            outcome = "CLEAN" if best == tid else "SWAP"
            break
    w.end_frame = last
    w.outcome = outcome
    if outcome == "SWAP":
        # swap frame = first frame after the last time the mask overlapped the seed track
        # at least as much as the settled track (robust to the ambiguous overlap period)
        b = int(w.settled_id)
        s = None
        for f, d in reversed(io_hist):
            if d.get(b, 0.0) <= d.get(tid, 0.0):
                break
            s = f
        if s is None:  # fallback: start of the final run of the settled id
            for f, a_id in reversed(assigned):
                if a_id is None:
                    continue
                if a_id != b:
                    break
                s = f
        w.swap_frame = s
    return w


def apply_swaps(tracks: np.ndarray, windows: list[Window], dedup_tol: int = 5) -> np.ndarray:
    """Relabel identities from each SWAP frame (A <-> B), in chronological order.

    The window seeded on A and the (witness) window seeded on B usually report the
    same event (A->B and B->A at ~the same frame); it is applied once. Windows that
    really changed the tracks get `applied = True` (duplicates stay False).
    """
    out = tracks.copy()
    perm: dict[int, int] = {}
    applied: list[tuple[frozenset, int, int, int]] = []  # pair, swap frame, window start, window end
    swaps = sorted([w for w in windows if w.outcome == "SWAP" and w.swap_frame is not None],
                   key=lambda w: w.swap_frame)
    for w in swaps:
        a, b = w.track_id, int(w.settled_id)
        key = frozenset((a, b))
        lo, hi = (w.seed_frame or w.swap_frame), (w.end_frame or w.swap_frame)
        if any(k == key and (abs(s - w.swap_frame) <= dedup_tol or (lo <= e and s0 <= hi))
               for k, s, s0, e in applied):
            continue  # same event already applied (symmetric or overlapping window)
        applied.append((key, w.swap_frame, lo, hi))
        la, lb = perm.get(a, a), perm.get(b, b)
        if la == lb:
            continue
        w.applied = True
        sel = out[:, F] >= w.swap_frame
        ia = sel & (out[:, ID] == la)
        ib = sel & (out[:, ID] == lb)
        out[ia, ID] = lb
        out[ib, ID] = la
        perm[a], perm[b] = lb, la
    return out


def run_smp(tracks: np.ndarray, log: pd.DataFrame, backend, triggers: pd.DataFrame,
            seq_len: int, params: Optional[SMPParams] = None) -> tuple[np.ndarray, list[Window], dict]:
    """Open windows for `triggers` (frame, track_id[, trigger]) and correct the tracks."""
    p = params or SMPParams()
    idx = _Index(tracks, log)
    covered: dict[int, int] = {}
    windows: list[Window] = []
    trig = triggers.sort_values(["frame", "track_id"]) if len(triggers) else triggers
    for row in trig.itertuples(index=False):
        t, tid = int(row.frame), int(row.track_id)
        kind = getattr(row, "trigger", "external")
        if covered.get(tid, -1) >= t:
            continue
        targets = [(tid, None)]
        if p.witness and tid in idx.track_box and t in idx.track_box[tid]:
            ids, boxes = idx.frame(t)
            ov = iou_matrix(idx.track_box[tid][t][None], boxes)[0] if len(boxes) else np.zeros(0)
            targets += [(int(i), tid) for i, o in zip(ids, ov) if int(i) != tid and o > p.witness_iou]
        for target, witness_of in targets:
            if covered.get(target, -1) >= t:
                continue
            w = _run_window(idx, backend, Window(target, t, kind if witness_of is None else "witness",
                                                  witness_of), seq_len, p)
            windows.append(w)
            covered[target] = max(w.end_frame or t, t)
    corrected = apply_swaps(tracks, windows)
    outcomes = Counter(w.outcome for w in windows)
    n_wit = sum(w.witness_of is not None for w in windows)
    stats = dict(windows=len(windows), vos_frames=int(sum(w.frames for w in windows)),
                 triggers=int(len(triggers)), windows_trigger=len(windows) - n_wit, windows_witness=n_wit,
                 windows_ran=sum(w.outcome != "SEED_FAIL" for w in windows),
                 swaps_applied=sum(w.applied for w in windows),
                 **{f"n_{k}": v for k, v in sorted(outcomes.items())})
    return corrected, windows, stats


def windows_frame(windows: list[Window]) -> pd.DataFrame:
    return pd.DataFrame([asdict(w) for w in windows])
