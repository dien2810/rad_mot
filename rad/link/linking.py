"""Learned tracklet linking (method 3) and VOS verification of uncertain links (method 4).

A fragment is an object the tracker loses as track A and restarts later as a new track C.
Fixing it needs no mask propagation: C (and whatever follows it) is relabelled as A.
For every pair (A ended, C born soon after, near where A was heading) a classifier gives
p = P(A and C are the same object), learned from GT on other videos, and

  * p >= tau_high            -> merge directly, no SAM call (method 3);
  * tau_low <= p < tau_high  -> ask a VOS model: seed A's last clean box, propagate to C's
                                start, merge only if the mask lands on C (method 4);
  * otherwise                -> leave the tracks as they are.

Merges are one-to-one (Hungarian on log-odds): an ended track gets at most one successor
and a new track at most one predecessor, so merged tracks never overlap in time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from rad.boxes import iou_matrix
from rad.io.mot import F, ID, X, Y, W, H, S, C
from rad.sam.masks import ioma

LINK_FEATURES = [
    "dt", "dt_s", "dist_ext", "dist_last", "iou_last", "iou_ext", "log_size_ratio", "log_aspect_ratio",
    "speed_a", "speed_c", "vel_cos", "vel_diff", "len_a", "len_c", "score_a", "score_c",
    "a_end_border", "c_start_border", "same_cls", "n_a_for_c", "n_c_for_a", "rank_for_c", "rank_for_a",
    "density",
]


@dataclass
class LinkParams:
    max_gap_s: float = 3.0       # seconds between A's last frame and C's first frame
    radius: float = 5.0          # keep pairs whose distance (extrapolated or last) <= radius * size of A
    tail: int = 10               # frames of A's end used for its velocity
    head: int = 10               # frames of C's start used for its velocity
    border: float = 0.01         # a box within this fraction of the image size touches the border
    label_k: int = 10            # matched frames used to read the GT identity of a track end/start
    # verification (method 4)
    seed_lookback: int = 15      # look this many frames back from A's end for a clean seed box
    seed_iso_iou: float = 0.3    # a seed box overlapping another track more than this is not clean
    check_len: int = 5           # frames of C checked against the mask
    max_verify_len: int = 150    # longest propagation allowed for one verification
    confirm_ioma: float = 0.5    # median share of the mask inside C's box needed to confirm


def _xyxy(r: np.ndarray) -> np.ndarray:
    r = np.atleast_2d(r)
    return np.stack([r[:, X], r[:, Y], r[:, X] + r[:, W], r[:, Y] + r[:, H]], 1)


def _slope(t: np.ndarray, v: np.ndarray) -> float:
    return float(np.polyfit(t, v, 1)[0]) if len(t) > 1 and np.ptp(t) > 0 else 0.0


def track_summary(tracks: np.ndarray, size: tuple, p: LinkParams) -> pd.DataFrame:
    """One row per track: first/last box, velocities at both ends, length, score, class, border flags."""
    width, height = size
    rows = []
    order = np.lexsort((tracks[:, F], tracks[:, ID]))
    t = tracks[order]
    ids, starts = np.unique(t[:, ID].astype(int), return_index=True)
    bounds = list(starts) + [len(t)]
    for k, tid in enumerate(ids):
        r = t[bounds[k]:bounds[k + 1]]
        cx, cy = r[:, X] + r[:, W] / 2, r[:, Y] + r[:, H] / 2
        tl, hd = r[-p.tail:], r[:p.head]
        tcx, tcy = cx[-p.tail:], cy[-p.tail:]
        hcx, hcy = cx[:p.head], cy[:p.head]
        first, last = r[0], r[-1]

        def at_border(b):
            mx, my = p.border * width, p.border * height
            return float(b[X] <= mx or b[Y] <= my or b[X] + b[W] >= width - mx or b[Y] + b[H] >= height - my)

        cls = r[:, C][r[:, C] >= 0]
        rows.append(dict(
            id=int(tid), start=int(first[F]), end=int(last[F]), length=len(r),
            first_x=first[X], first_y=first[Y], first_w=first[W], first_h=first[H],
            last_x=last[X], last_y=last[Y], last_w=last[W], last_h=last[H],
            vx_end=_slope(tl[:, F], tcx), vy_end=_slope(tl[:, F], tcy),
            vx_start=_slope(hd[:, F], hcx), vy_start=_slope(hd[:, F], hcy),
            score_end=float(np.mean(tl[:, S])), score_start=float(np.mean(hd[:, S])),
            cls=int(pd.Series(cls).mode().iloc[0]) if len(cls) else -1,
            end_border=at_border(last), start_border=at_border(first)))
    return pd.DataFrame(rows).set_index("id") if rows else pd.DataFrame()


def link_candidates(tracks: np.ndarray, size: tuple, fps: float, p: LinkParams | None = None) -> pd.DataFrame:
    """All (A ended, C born) pairs that could be the same object, with LINK_FEATURES."""
    p = p or LinkParams()
    cols = ["a", "c", "a_end", "c_start"] + LINK_FEATURES
    if len(tracks) == 0:
        return pd.DataFrame(columns=cols)
    fps = float(fps or 30.0)
    max_gap = max(1, int(round(p.max_gap_s * fps)))
    ts = track_summary(tracks, size, p)
    first_frame = int(tracks[:, F].min())
    dens = pd.Series(tracks[:, F]).value_counts()
    rows = []
    for cid, c in ts[ts.start > first_frame].iterrows():
        a_all = ts[(ts.end < c.start) & (ts.end >= c.start - max_gap) & (ts.index != cid)]
        if len(a_all) == 0:
            continue
        dt = (c.start - a_all.end).to_numpy(float)
        size_a = np.sqrt(a_all.last_w * a_all.last_h).clip(lower=1.0).to_numpy()
        acx = (a_all.last_x + a_all.last_w / 2).to_numpy()
        acy = (a_all.last_y + a_all.last_h / 2).to_numpy()
        ex, ey = acx + a_all.vx_end.to_numpy() * dt, acy + a_all.vy_end.to_numpy() * dt
        ccx, ccy = c.first_x + c.first_w / 2, c.first_y + c.first_h / 2
        d_ext = np.hypot(ex - ccx, ey - ccy) / size_a
        d_last = np.hypot(acx - ccx, acy - ccy) / size_a
        keep = (d_ext <= p.radius) | (d_last <= p.radius)
        if not keep.any():
            continue
        cbox = np.array([[c.first_x, c.first_y, c.first_x + c.first_w, c.first_y + c.first_h]])
        for j in np.flatnonzero(keep):
            a = a_all.iloc[j]
            abox = np.array([[a.last_x, a.last_y, a.last_x + a.last_w, a.last_y + a.last_h]])
            shift = np.array([[a.vx_end * dt[j], a.vy_end * dt[j]] * 2])
            va, vc = np.array([a.vx_end, a.vy_end]), np.array([c.vx_start, c.vy_start])
            na, nc = np.linalg.norm(va), np.linalg.norm(vc)
            rows.append(dict(
                a=int(a.name), c=int(cid), a_end=int(a.end), c_start=int(c.start),
                dt=float(dt[j]), dt_s=float(dt[j] / fps), dist_ext=float(d_ext[j]), dist_last=float(d_last[j]),
                iou_last=float(iou_matrix(abox, cbox)[0, 0]), iou_ext=float(iou_matrix(abox + shift, cbox)[0, 0]),
                log_size_ratio=float(np.log(max(np.sqrt(c.first_w * c.first_h), 1.0) / size_a[j])),
                log_aspect_ratio=float(np.log(max(c.first_w / max(c.first_h, 1.0), 1e-3)
                                              / max(a.last_w / max(a.last_h, 1.0), 1e-3))),
                speed_a=float(na / size_a[j]), speed_c=float(nc / size_a[j]),
                vel_cos=float(va @ vc / (na * nc)) if na > 1e-6 and nc > 1e-6 else 0.0,
                vel_diff=float(np.linalg.norm(va - vc) / size_a[j]),
                len_a=float(a.length), len_c=float(c.length), score_a=float(a.score_end), score_c=float(c.score_start),
                a_end_border=float(a.end_border), c_start_border=float(c.start_border),
                same_cls=float(a.cls == c.cls), density=float(dens.get(int(c.start), 0))))
    out = pd.DataFrame(rows, columns=cols)
    if len(out):
        out["n_a_for_c"] = out.groupby("c").a.transform("size").astype(float)
        out["n_c_for_a"] = out.groupby("a").c.transform("size").astype(float)
        out["rank_for_c"] = out.groupby("c").dist_ext.rank(method="first").astype(float)
        out["rank_for_a"] = out.groupby("a").dist_ext.rank(method="first").astype(float)
    return out


def track_identities(matches: pd.DataFrame, k: int) -> pd.DataFrame:
    """GT identity at each track's end and start: the most common GT id in its last/first k matched frames."""
    if len(matches) == 0:
        return pd.DataFrame(columns=["gid_end", "gid_start"])
    m = matches.sort_values(["trk_id", "frame"])
    mode = lambda s: int(s.mode().iloc[0]) if len(s) else -1  # noqa: E731
    end = m.groupby("trk_id").tail(k).groupby("trk_id").gt_id.agg(mode)
    start = m.groupby("trk_id").head(k).groupby("trk_id").gt_id.agg(mode)
    return pd.DataFrame({"gid_end": end, "gid_start": start})


def label_links(cands: pd.DataFrame, matches: pd.DataFrame, k: int = 10) -> np.ndarray:
    """1 if A's end and C's start are matched to the same GT object, else 0."""
    if len(cands) == 0:
        return np.zeros(0, dtype=int)
    ident = track_identities(matches, k)
    ge = cands.a.map(ident.gid_end).fillna(-1).astype(int).to_numpy()
    gs = cands.c.map(ident.gid_start).fillna(-1).astype(int).to_numpy()
    return ((ge == gs) & (ge >= 0)).astype(int)


def assign(cands: pd.DataFrame, score: str, tau: float) -> pd.DataFrame:
    """One-to-one choice among pairs with score >= tau, maximising the summed log-odds."""
    sel = cands[cands[score] >= tau].drop_duplicates(["a", "c"])
    if len(sel) == 0:
        return sel
    a_ids, ai = np.unique(sel.a.to_numpy(), return_inverse=True)
    c_ids, ci = np.unique(sel.c.to_numpy(), return_inverse=True)
    na = len(a_ids)
    # pairs are local in time and space, so the bipartite graph splits into small components:
    # solve each one exactly instead of one huge assignment problem
    adj = coo_matrix((np.ones(len(sel)), (ai, na + ci)), shape=(na + len(c_ids),) * 2)
    _, comp = connected_components(adj, directed=False)
    pr = sel[score].clip(1e-6, 1 - 1e-6).to_numpy()
    gain = np.log(pr / (1 - pr)) + 50.0  # shift: every real pair beats "no pair"
    keep = np.zeros(len(sel), dtype=bool)
    pair_comp = comp[ai]
    order = np.argsort(pair_comp, kind="stable")
    bounds = np.flatnonzero(np.diff(pair_comp[order])) + 1
    for idx in np.split(order, bounds):
        if len(idx) == 1:
            keep[idx] = True
            continue
        ra, rinv = np.unique(ai[idx], return_inverse=True)
        rc, cinv = np.unique(ci[idx], return_inverse=True)
        cost = np.full((len(ra), len(rc)), 1e6)
        cost[rinv, cinv] = -gain[idx]
        r, c = linear_sum_assignment(cost)
        ok = cost[r, c] < 1e5
        chosen = set(zip(r[ok], c[ok]))
        keep[idx] = [(x, y) in chosen for x, y in zip(rinv, cinv)]
    return sel[keep]


def apply_links(tracks: np.ndarray, pairs: pd.DataFrame) -> np.ndarray:
    """Relabel each C (and its own successors) with the id of the start of its chain."""
    out = tracks.copy()
    if len(pairs) == 0:
        return out
    parent = dict(zip(pairs.c.astype(int), pairs.a.astype(int)))

    def root(x: int) -> int:
        seen = set()
        while x in parent and x not in seen:
            seen.add(x)
            x = parent[x]
        return x

    ids = out[:, ID].astype(int)
    out[:, ID] = [root(i) if i in parent else i for i in ids]
    return out


def verify_pair(backend, tracks: np.ndarray, a: int, c: int, p: LinkParams | None = None) -> dict:
    """Seed the VOS model on A's last clean box and check whether the mask lands on C."""
    p = p or LinkParams()
    ta = tracks[tracks[:, ID] == a]
    tc = tracks[tracks[:, ID] == c]
    ta, tc = ta[np.argsort(ta[:, F])], tc[np.argsort(tc[:, F])]
    seed = ta[-1]
    for r in ta[::-1][:p.seed_lookback]:
        f = r[F]
        others = tracks[(tracks[:, F] == f) & (tracks[:, ID] != a)]
        if len(others) == 0 or iou_matrix(_xyxy(r), _xyxy(others))[0].max() < p.seed_iso_iou:
            seed = r
            break
    seed_f, c_start = int(seed[F]), int(tc[0, F])
    check = tc[:p.check_len]
    last_check = int(check[-1, F])
    n = last_check - seed_f + 1
    info = dict(a=a, c=c, seed_frame=seed_f, frames=0, ioma_c=np.nan, ioma_other=np.nan, confirmed=False)
    if n > p.max_verify_len:
        info["skipped"] = "too_long"
        return info
    c_by_f = {int(r[F]): r for r in check}
    vals_c, vals_o = [], []
    for f, mask, _score in backend.propagate(seed_f, _xyxy(seed)[0], n):
        info["frames"] += 1
        if f in c_by_f:
            others = tracks[(tracks[:, F] == f) & (tracks[:, ID] != c) & (tracks[:, ID] != a)]
            vals_c.append(float(ioma(mask, _xyxy(c_by_f[f]))[0]))
            vals_o.append(float(ioma(mask, _xyxy(others)).max()) if len(others) else 0.0)
        if f >= last_check:
            break
    if len(vals_c) >= 2:
        mc, mo = float(np.median(vals_c)), float(np.median(vals_o))
        info.update(ioma_c=mc, ioma_other=mo, confirmed=bool(mc >= p.confirm_ioma and mc > mo))
    return info
