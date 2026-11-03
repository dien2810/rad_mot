"""Dispatch policies: which candidate events open a SAM window.

All policies return a DataFrame (frame, track_id, trigger) for rad.sam.smp.run_smp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rad.sam.smp import SMPParams, smp_triggers

_COLS = ["frame", "track_id", "trigger"]


def _fmt(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=_COLS)
    out = df[["frame", "track_id"]].copy()
    out["trigger"] = name
    return out.sort_values(["frame", "track_id"]).drop_duplicates(["frame", "track_id"]).reset_index(drop=True)


def smp_baseline(log: pd.DataFrame, tau: float = 0.05, gap_G: int = 5) -> pd.DataFrame:
    return smp_triggers(log, SMPParams(tau_entry=tau, gap_G=gap_G))


def smp_plus_births(log: pd.DataFrame, cands: pd.DataFrame, tau: float = 0.05, gap_G: int = 5) -> pd.DataFrame:
    base = smp_baseline(log, tau, gap_G)
    births = _fmt(cands[cands.kind == "birth"], "birth")
    return pd.concat([base, births]).sort_values(["frame", "track_id"]).drop_duplicates(
        ["frame", "track_id"]).reset_index(drop=True)


def density_rule(cands: pd.DataFrame, tau: float = 0.05, min_density: int = 2, gap_G: int = 5) -> pd.DataFrame:
    """Heuristic: low margin only counts in locally crowded spots (SAMIDARE-like idea)."""
    m = cands[(cands.kind == "match") & (((cands.margin_det < tau) & (cands.local_density >= min_density))
                                        | (cands.gap >= gap_G))]
    return _fmt(m, "density")


def threshold(cands: pd.DataFrame, score_col: str, theta: float) -> pd.DataFrame:
    return _fmt(cands[cands[score_col] >= theta], f"{score_col}>={theta:g}")


def budget(cands: pd.DataFrame, score_col: str, n: int) -> pd.DataFrame:
    """Top-n candidates by score (per call; call once per sequence)."""
    n = int(max(n, 0))
    top = cands.sort_values(score_col, ascending=False).head(n)
    return _fmt(top, f"{score_col}@{n}")


def random_budget(cands: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = int(min(max(n, 0), len(cands)))
    pick = cands.iloc[rng.choice(len(cands), size=n, replace=False)] if n else cands.iloc[:0]
    return _fmt(pick, f"random@{n}")
