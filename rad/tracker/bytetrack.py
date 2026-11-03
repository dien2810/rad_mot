"""ByteTrack (Zhang et al., ECCV 2022) re-implemented with association logging.

The log records, for every matched (track, detection) pair, the quantities that
Selective Mask Propagation uses to decide when to call SAM (assignment margin,
track gap) plus cheap context features (local density, overlaps, Kalman
innovation, ...). These rows are the "candidate events" of the project.

Margin follows SMP: for a matched detection j* assigned to track i*,
    margin_det = min_{i != i*} C[i, j*] - C[i*, j*]
(second-best competing track minus the chosen one; small = ambiguous).
margin_trk is the symmetric quantity over detections for track i*.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from rad.boxes import centers, iou_matrix
from rad.io.mot import F, S, C as CLS_COL, X, Y, W, H, group_by_frame
from rad.tracker.kalman import KalmanFilter

NEW, TRACKED, LOST, REMOVED = 0, 1, 2, 3


@dataclass
class ByteTrackParams:
    track_thresh: float = 0.5     # high-score detections
    low_thresh: float = 0.1       # low-score detections (second association)
    new_track_offset: float = 0.1  # new tracks need score >= track_thresh + offset
    match_thresh: float = 0.8     # first association cost threshold
    second_thresh: float = 0.5
    unconfirmed_thresh: float = 0.7
    track_buffer: int = 30
    frame_rate: float = 30.0
    fuse_score: bool = True
    class_aware: bool = False
    min_box_area: float = 10.0
    log_density_radius: float = 2.0  # radius = factor * max(w, h)


def _tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
    ret = np.asarray(tlwh, dtype=float).copy()
    ret[:2] += ret[2:] / 2
    ret[2] /= max(ret[3], 1e-6)
    return ret


class STrack:
    def __init__(self, tlwh, score: float, cls: int):
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None
        self.is_activated = False
        self.score = float(score)
        self.cls_votes: dict[int, float] = {int(cls): float(score)}
        self.tracklet_len = 0
        self.state = NEW
        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0
        self.det_index = -1

    @property
    def cls(self) -> int:
        return max(self.cls_votes.items(), key=lambda kv: kv[1])[0]

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        ret = self.tlwh
        ret[2:] += ret[:2]
        return ret

    def predict(self, kf: KalmanFilter) -> None:
        mean_state = self.mean.copy()
        if self.state != TRACKED:
            mean_state[7] = 0
        self.mean, self.covariance = kf.predict(mean_state, self.covariance)

    def activate(self, kf: KalmanFilter, frame_id: int, track_id: int) -> None:
        self.track_id = track_id
        self.mean, self.covariance = kf.initiate(_tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TRACKED
        self.is_activated = frame_id == 1
        self.frame_id = frame_id
        self.start_frame = frame_id

    def _absorb(self, det: "STrack", kf: KalmanFilter, frame_id: int) -> None:
        self.mean, self.covariance = kf.update(self.mean, self.covariance, _tlwh_to_xyah(det.tlwh))
        self.state = TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.score = det.score
        self.det_index = det.det_index
        for c, v in det.cls_votes.items():
            self.cls_votes[c] = self.cls_votes.get(c, 0.0) + v

    def re_activate(self, det: "STrack", kf: KalmanFilter, frame_id: int) -> None:
        self._absorb(det, kf, frame_id)
        self.tracklet_len = 0

    def update(self, det: "STrack", kf: KalmanFilter, frame_id: int) -> None:
        self.tracklet_len += 1
        self._absorb(det, kf, frame_id)


def _assign(cost: np.ndarray, thresh: float):
    """Optimal assignment keeping only pairs with cost <= thresh (lapjv cost_limit semantics)."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    big = thresh + 1e5
    c = np.where(cost > thresh, big, cost)
    rows, cols = linear_sum_assignment(c)
    matches = [(r, k) for r, k in zip(rows, cols) if cost[r, k] <= thresh]
    mr = {r for r, _ in matches}
    mc = {k for _, k in matches}
    return (matches, [r for r in range(cost.shape[0]) if r not in mr],
            [k for k in range(cost.shape[1]) if k not in mc])


def _joint(a: list, b: list) -> list:
    seen = {t.track_id for t in a}
    return a + [t for t in b if t.track_id not in seen]


def _sub(a: list, b: list) -> list:
    ids = {t.track_id for t in b}
    return [t for t in a if t.track_id not in ids]


def _remove_duplicates(a: list, b: list):
    if not a or not b:
        return a, b
    ious = iou_matrix(np.array([t.tlbr for t in a]), np.array([t.tlbr for t in b]))
    pa, pb = np.where(ious > 0.85)
    dup_a, dup_b = set(), set()
    for p, q in zip(pa, pb):
        ta = a[p].frame_id - a[p].start_frame
        tb = b[q].frame_id - b[q].start_frame
        if ta > tb:
            dup_b.add(q)
        else:
            dup_a.add(p)
    return [t for i, t in enumerate(a) if i not in dup_a], [t for i, t in enumerate(b) if i not in dup_b]


class BYTETracker:
    def __init__(self, params: Optional[ByteTrackParams] = None):
        self.p = params or ByteTrackParams()
        self.kf = KalmanFilter()
        self.tracked: list[STrack] = []
        self.lost: list[STrack] = []
        self.frame_id = 0
        self.max_time_lost = int(self.p.frame_rate / 30.0 * self.p.track_buffer)
        self._next_id = 0
        self.log_rows: list[dict] = []

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    # ------------------------------------------------------------------ costs
    def _cost(self, tracks: list, dets: list, fuse: bool):
        if not tracks or not dets:
            z = np.zeros((len(tracks), len(dets)))
            return z + 1.0, z
        iou = iou_matrix(np.array([t.tlbr for t in tracks]), np.array([d.tlbr for d in dets]))
        sim = iou.copy()
        if fuse:
            sim = sim * np.array([d.score for d in dets])[None, :]
        cost = 1.0 - sim
        if self.p.class_aware:
            tc = np.array([t.cls for t in tracks])[:, None]
            dc = np.array([d.cls for d in dets])[None, :]
            cost = np.where(tc == dc, cost, 1.0)
        return cost, iou

    # ---------------------------------------------------------------- logging
    def _log(self, stage: int, cost: np.ndarray, iou: np.ndarray, matches, tracks, dets,
             frame_ctx: dict) -> None:
        for r, k in matches:
            t, d = tracks[r], dets[k]
            col = np.delete(cost[:, k], r)
            row = np.delete(cost[r, :], k)
            c = float(cost[r, k])
            margin_det = float(col.min() - c) if col.size else 1.0
            margin_trk = float(row.min() - c) if row.size else 1.0
            dbox = d.tlbr
            j = d.det_index
            ctx_iou = frame_ctx["iou"][j]
            others = np.delete(ctx_iou, j)
            dist = np.linalg.norm(frame_ctx["ctr"] - frame_ctx["ctr"][j], axis=1)
            radius = self.p.log_density_radius * max(dbox[2] - dbox[0], dbox[3] - dbox[1])
            pred = t.tlwh
            dw, dh = dbox[2] - dbox[0], dbox[3] - dbox[1]
            self.log_rows.append(dict(
                frame=self.frame_id, track_id=t.track_id, det_index=j, stage=stage,
                cost=c, margin_det=margin_det, margin_trk=margin_trk,
                margin=min(margin_det, margin_trk), iou_pred=float(iou[r, k]),
                n_track_candidates=int((iou[:, k] > 0).sum()),
                n_det_candidates=int((iou[r, :] > 0).sum()),
                gap=int(self.frame_id - t.frame_id), was_lost=int(t.state == LOST),
                tracklet_len=int(t.tracklet_len), track_age=int(self.frame_id - t.start_frame),
                det_score=float(d.score), det_cls=int(d.cls), track_cls=int(t.cls),
                innovation=self.kf.mahalanobis(t.mean, t.covariance, _tlwh_to_xyah(d.tlwh)),
                speed=float(np.hypot(t.mean[4], t.mean[5])),
                aspect_ratio_change=float((dw / max(dh, 1e-6)) / max(pred[2] / max(pred[3], 1e-6), 1e-6)),
                height_ratio=float(dh / max(pred[3], 1e-6)),
                x=float(dbox[0]), y=float(dbox[1]), w=float(dw), h=float(dh),
                n_overlap_dets=int((others > 0).sum()),
                max_iou_other_det=float(others.max()) if others.size else 0.0,
                local_density=int((dist < radius).sum() - 1),
                frame_density=int(len(frame_ctx["ctr"])),
            ))

    # ----------------------------------------------------------------- update
    def update(self, dets: np.ndarray) -> list[STrack]:
        """dets: (N, 6) array [x1, y1, x2, y2, score, cls]."""
        self.frame_id += 1
        p = self.p
        dets = np.asarray(dets, dtype=float).reshape(-1, 6)
        keep = dets[:, 4] > p.low_thresh
        dets = dets[keep]
        frame_ctx = {"iou": iou_matrix(dets[:, :4], dets[:, :4]), "ctr": centers(dets[:, :4])}

        def mk(idx):
            out = []
            for j in idx:
                x1, y1, x2, y2, s, c = dets[j]
                st = STrack([x1, y1, x2 - x1, y2 - y1], s, int(c))
                st.det_index = int(j)
                out.append(st)
            return out

        high = mk(np.flatnonzero(dets[:, 4] >= p.track_thresh))
        low = mk(np.flatnonzero(dets[:, 4] < p.track_thresh))

        unconfirmed = [t for t in self.tracked if not t.is_activated]
        tracked = [t for t in self.tracked if t.is_activated]
        activated, refind, lost, removed = [], [], [], []

        # Step 2: first association with high-score detections
        pool = _joint(tracked, self.lost)
        for t in pool:
            t.predict(self.kf)
        cost, iou = self._cost(pool, high, p.fuse_score)
        matches, u_track, u_det = _assign(cost, p.match_thresh)
        self._log(1, cost, iou, matches, pool, high, frame_ctx)
        for r, k in matches:
            t, d = pool[r], high[k]
            if t.state == TRACKED:
                t.update(d, self.kf, self.frame_id)
                activated.append(t)
            else:
                t.re_activate(d, self.kf, self.frame_id)
                refind.append(t)

        # Step 3: second association with low-score detections
        r_tracked = [pool[i] for i in u_track if pool[i].state == TRACKED]
        cost2, iou2 = self._cost(r_tracked, low, False)
        matches2, u_track2, _ = _assign(cost2, p.second_thresh)
        self._log(2, cost2, iou2, matches2, r_tracked, low, frame_ctx)
        for r, k in matches2:
            t, d = r_tracked[r], low[k]
            if t.state == TRACKED:
                t.update(d, self.kf, self.frame_id)
                activated.append(t)
            else:  # pragma: no cover - r_tracked only holds TRACKED
                t.re_activate(d, self.kf, self.frame_id)
                refind.append(t)
        for i in u_track2:
            t = r_tracked[i]
            if t.state != LOST:
                t.state = LOST
                lost.append(t)

        # Unconfirmed tracks (seen once) vs remaining high detections
        rem = [high[k] for k in u_det]
        cost3, iou3 = self._cost(unconfirmed, rem, p.fuse_score)
        matches3, u_unconf, u_det3 = _assign(cost3, p.unconfirmed_thresh)
        self._log(3, cost3, iou3, matches3, unconfirmed, rem, frame_ctx)
        for r, k in matches3:
            unconfirmed[r].update(rem[k], self.kf, self.frame_id)
            activated.append(unconfirmed[r])
        for i in u_unconf:
            unconfirmed[i].state = REMOVED
            removed.append(unconfirmed[i])

        # Step 4: new tracks
        for k in u_det3:
            d = rem[k]
            if d.score < p.track_thresh + p.new_track_offset:
                continue
            if d.tlwh[2] * d.tlwh[3] < p.min_box_area:
                continue
            d.activate(self.kf, self.frame_id, self._new_id())
            activated.append(d)

        # Step 5: drop old lost tracks
        for t in self.lost:
            if self.frame_id - t.frame_id > self.max_time_lost:
                t.state = REMOVED
                removed.append(t)

        self.tracked = [t for t in self.tracked if t.state == TRACKED]
        self.tracked = _joint(self.tracked, activated)
        self.tracked = _joint(self.tracked, refind)
        self.lost = _sub(self.lost, self.tracked)
        self.lost.extend(lost)
        self.lost = _sub(self.lost, removed)
        self.tracked, self.lost = _remove_duplicates(self.tracked, self.lost)
        return [t for t in self.tracked if t.is_activated]


def run_bytetrack(det: np.ndarray, n_frames: Optional[int] = None,
                  params: Optional[ByteTrackParams] = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Run ByteTrack on detections in the internal 9-column layout.

    Returns (tracks (N, 9) with score and majority class, association log DataFrame).
    """
    params = params or ByteTrackParams()
    tracker = BYTETracker(params)
    by_frame = group_by_frame(det)
    if n_frames is None:
        n_frames = int(det[:, F].max()) if len(det) else 0
    out = []
    for frame in range(1, n_frames + 1):
        rows = by_frame.get(frame)
        if rows is None:
            d = np.zeros((0, 6))
        else:
            d = np.stack([rows[:, X], rows[:, Y], rows[:, X] + rows[:, W], rows[:, Y] + rows[:, H],
                          rows[:, S], rows[:, CLS_COL]], axis=1)
        for t in tracker.update(d):
            tlwh = t.tlwh
            if tlwh[2] * tlwh[3] < params.min_box_area:
                continue
            out.append([frame, t.track_id, *tlwh, t.score, t.cls, 1.0])
    tracks = np.asarray(out, dtype=float) if out else np.zeros((0, 9))
    log = pd.DataFrame(tracker.log_rows)
    return tracks, log


def gt_as_detections(gt: np.ndarray, vis_min: float = 0.0, score: float = 0.95,
                     drop_rate: float = 0.0, jitter: float = 0.0, seed: int = 0) -> np.ndarray:
    """Oracle detections from GT (optionally noisy) to study association in isolation.

    Detections with visibility < vis_min are dropped; low visibility lowers the score so
    that occluded objects fall into ByteTrack's low-score stage.
    """
    rng = np.random.default_rng(seed)
    d = gt.copy()
    keep = d[:, 8] >= vis_min
    if drop_rate > 0:
        keep &= rng.random(len(d)) >= drop_rate
    d = d[keep]
    if jitter > 0:
        d[:, X] += rng.normal(0, jitter * d[:, W])
        d[:, Y] += rng.normal(0, jitter * d[:, H])
        d[:, W] *= np.exp(rng.normal(0, jitter, len(d)))
        d[:, H] *= np.exp(rng.normal(0, jitter, len(d)))
    d[:, S] = np.clip(score * (0.4 + 0.6 * np.clip(d[:, 8], 0, 1)), 0.05, 0.99)
    d[:, 1] = -1
    return d
