"""Candidate events and cheap features for the learned dispatch policy.

Two kinds of candidate (frame, track_id):
  match  - a matched (track, detection) pair from the ByteTrack log, kept if it looks even
           slightly ambiguous (relaxed filter). Same signals SMP uses, plus context.
  birth  - a new track C is born near where a track A was recently lost. SMP's gap
           trigger only fires when A itself is re-found; if the object got a fresh ID
           instead (fragmentation), SMP never looks. A window seeded on A can relabel C.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rad.io.mot import F

MATCH_FEATURES = [
    "margin_det", "margin_trk", "margin", "cost", "iou_pred", "n_track_candidates", "n_det_candidates",
    "gap", "was_lost", "tracklet_len", "track_age", "det_score", "innovation", "speed",
    "aspect_ratio_change", "height_ratio", "w", "h", "n_overlap_dets", "max_iou_other_det",
    "local_density", "frame_density", "stage",
    "min_margin_5", "n_low_margin_10", "max_overlap_5", "frames_since_low",
]
BIRTH_FEATURES = ["birth_dist", "birth_dt", "birth_size_ratio", "birth_n_partners"]
FEATURES = MATCH_FEATURES + BIRTH_FEATURES + ["is_birth"]


@dataclass
class CandidateParams:
    relax_margin: float = 0.3
    low_margin: float = 0.1
    birth_window: int = 30        # frames: A lost at most this long before C is born
    birth_radius: float = 3.0     # distance in units of A's box size
    include_births: bool = True


def _rolling(log: pd.DataFrame, low_margin: float) -> pd.DataFrame:
    lg = log.sort_values(["track_id", "frame"]).copy()
    g = lg.groupby("track_id", sort=False)
    lg["min_margin_5"] = g["margin_det"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).min())
    low = (lg["margin_det"] < low_margin).astype(float)
    lg["_low"] = low
    lg["n_low_margin_10"] = g["_low"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).sum())
    lg["max_overlap_5"] = g["n_overlap_dets"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).max())
    last_low = lg["frame"].where(low.astype(bool))
    lg["_last_low"] = last_low.groupby(lg["track_id"]).transform(lambda s: s.shift(1).ffill())
    lg["frames_since_low"] = (lg["frame"] - lg["_last_low"]).fillna(999)
    return lg.drop(columns=["_low", "_last_low"])


def match_candidates(log: pd.DataFrame, p: CandidateParams) -> pd.DataFrame:
    if len(log) == 0:
        return pd.DataFrame(columns=["frame", "track_id", "kind"] + FEATURES)
    lg = _rolling(log, p.low_margin)
    keep = ((lg.margin_det < p.relax_margin) | (lg.gap >= 2) | (lg.n_overlap_dets >= 1)
            | (lg.n_track_candidates >= 2))
    c = lg[keep].copy()
    c["kind"] = "match"
    c["partner"] = -1
    for col in BIRTH_FEATURES:
        c[col] = np.nan
    c["is_birth"] = 0
    return c


def birth_candidates(tracks: np.ndarray, p: CandidateParams) -> pd.DataFrame:
    """Pair every new track with recently ended tracks whose extrapolated box is nearby."""
    cols = ["frame", "track_id", "kind", "partner"] + BIRTH_FEATURES
    if len(tracks) == 0:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(tracks[:, :6], columns=["frame", "id", "x", "y", "w", "h"])
    df["cx"], df["cy"] = df.x + df.w / 2, df.y + df.h / 2
    df = df.sort_values(["id", "frame"])
    first = df.groupby("id").head(1).set_index("id")
    tails = df.groupby("id").tail(5)
    last = df.groupby("id").tail(1).set_index("id")
    vel = tails.groupby("id").apply(
        lambda t: pd.Series({"vx": np.polyfit(t.frame, t.cx, 1)[0] if len(t) > 1 else 0.0,
                             "vy": np.polyfit(t.frame, t.cy, 1)[0] if len(t) > 1 else 0.0}),
        include_groups=False)
    last = last.join(vel)
    first_frame = int(df.frame.min())
    rows = []
    for cid, c in first.iterrows():
        s = int(c.frame)
        if s <= first_frame:
            continue
        cand = last[(last.frame < s) & (last.frame >= s - p.birth_window) & (last.index != cid)]
        if len(cand) == 0:
            continue
        dt = s - cand.frame
        px, py = cand.cx + cand.vx * dt, cand.cy + cand.vy * dt
        size = np.sqrt(cand.w * cand.h).clip(lower=1.0)
        dist = np.hypot(px - c.cx, py - c.cy) / size
        near = cand[dist <= p.birth_radius]
        for aid, a in near.iterrows():
            rows.append(dict(frame=s, track_id=int(aid), kind="birth", partner=int(cid),
                             birth_dist=float(dist[aid]), birth_dt=int(s - a.frame),
                             birth_size_ratio=float(np.sqrt(c.w * c.h) / max(np.sqrt(a.w * a.h), 1.0)),
                             birth_n_partners=int(len(near))))
    return pd.DataFrame(rows, columns=cols)


def build_candidates(log: pd.DataFrame, tracks: np.ndarray, p: CandidateParams | None = None) -> pd.DataFrame:
    p = p or CandidateParams()
    m = match_candidates(log, p)
    parts = [m]
    if p.include_births:
        b = birth_candidates(tracks, p)
        if len(b):
            dens = pd.Series(tracks[:, F]).value_counts()
            b["frame_density"] = b.frame.map(dens).fillna(0)
            for col in MATCH_FEATURES:
                if col not in b:
                    b[col] = np.nan
            b["is_birth"] = 1
            parts.append(b)
    out = pd.concat([x for x in parts if len(x)], ignore_index=True) if any(len(x) for x in parts) else m
    for col in FEATURES:
        if col not in out:
            out[col] = np.nan
    return out.sort_values(["frame", "track_id"]).reset_index(drop=True)
