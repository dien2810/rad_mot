"""Automatic labels for candidate events (no manual annotation).

y_risk : a GT identity switch involving this track happens within ±horizon frames.
         Needs only GT, no VOS.
y_fix  : opening a SAM window at this candidate would remove a GT identity switch
         locally (computed by running one window with a VOS backend, e.g. the oracle).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from rad.eval.idsw import ClearResult, clear_match
from rad.sam.smp import SMPParams, Window, _Index, _run_window, apply_swaps


def label_risk(cands: pd.DataFrame, cm: ClearResult, horizon: int = 10) -> pd.Series:
    sw = cm.switches
    if len(cands) == 0:
        return pd.Series(dtype=int)
    if len(sw) == 0:
        return pd.Series(0, index=cands.index)
    by_track: dict[int, np.ndarray] = {}
    for col in ("prev_trk", "new_trk"):
        for tid, grp in sw.groupby(col):
            by_track.setdefault(int(tid), [])
            by_track[int(tid)] = np.concatenate([np.asarray(by_track[int(tid)], float), grp.frame.to_numpy(float)])
    y = np.zeros(len(cands), dtype=int)
    for i, (f, t, partner) in enumerate(zip(cands.frame.to_numpy(), cands.track_id.to_numpy(),
                                            cands.get("partner", pd.Series(-1, index=cands.index)).to_numpy())):
        for tid in (int(t), int(partner)):
            fr = by_track.get(tid)
            if fr is not None and np.any(np.abs(fr - f) <= horizon):
                y[i] = 1
                break
    return pd.Series(y, index=cands.index)


def label_fix(cands: pd.DataFrame, gt: np.ndarray, tracks: np.ndarray, log: pd.DataFrame, backend,
              seq_len: int, params: Optional[SMPParams] = None, dedup: int = 5,
              eval_margin: int = 60) -> pd.DataFrame:
    """Run one window per (deduplicated) candidate; label 1 if it lowers local IDSW.

    Returns DataFrame indexed like cands with y_fix, fix_gain, win_frames, win_outcome.
    """
    p = params or SMPParams()
    idx = _Index(tracks, log)
    out = pd.DataFrame(index=cands.index, data={"y_fix": 0, "fix_gain": 0, "win_frames": 0,
                                                "win_outcome": ""})
    last_run: dict[int, tuple[int, dict]] = {}
    for i, row in cands.sort_values("frame").iterrows():
        t, tid = int(row.frame), int(row.track_id)
        prev = last_run.get(tid)
        if prev is not None and t - prev[0] <= dedup:
            res = prev[1]
        else:
            w = _run_window(idx, backend, Window(tid, t, str(row.get("kind", "match"))), seq_len, p)
            gain = 0
            if w.outcome == "SWAP":
                lo = (w.seed_frame or t) - 1
                hi = (w.end_frame or t) + eval_margin
                before = clear_match(gt, tracks, frames=(lo, hi)).num_switches
                after = clear_match(gt, apply_swaps(tracks, [w]), frames=(lo, hi)).num_switches
                gain = before - after
            res = dict(y_fix=int(gain > 0), fix_gain=int(gain), win_frames=int(w.frames), win_outcome=w.outcome)
            last_run[tid] = (t, res)
        for k, v in res.items():
            out.at[i, k] = v
    return out
