"""Pilot analysis for hypotheses H1/H2 (and P4) of the project.

H1  The share of SAM windows that change nothing grows with density.
H2  A noticeable share of real identity switches is never covered by a window.
P4  Seed search fails more often in dense scenes.

Definitions used in every table (README, "Pilot H1/H2"):
- density  objects in a frame. Default: GT objects ("gt"), which is the same for every
           run (oracle / YOLO / SAM 3). Option: tracker boxes ("tracker"). H1 and H2 use
           the same measure, and frames without any object count as density 0.
- kind     'trigger' = opened by the margin/gap rule; 'witness' = opened for a track whose
           box overlaps a triggered track.
- ran      the seed search succeeded, so SAM/VOS actually processed frames.
           SEED_FAIL windows cost no SAM time and are kept apart.
- wasted   ran but changed no ID: outcome other than SWAP, or a SWAP that apply_swaps
           skipped as a duplicate (the A<->B event already applied by another window).
- coverage (switches) 'ran' = a window that ran spans the switch frame (+- slack) on the
           old or new track; 'seed_fail' = only a SEED_FAIL window was opened there;
           'none' = no window at all.

Inputs are produced by the tracker (association log), SMP (windows) and the GT
matcher (switch events), so the same code runs on synthetic and real data.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from rad.eval.idsw import ClearResult
from rad.io.mot import F, ID

DEFAULT_BINS = (0, 10, 20, 30, 40, 60, 80, 120, 10_000)
DENSITY_LABEL = {"gt": "GT objects per frame", "tracker": "tracker boxes per frame"}


def density_per_frame(arr: np.ndarray, length: Optional[int] = None) -> pd.Series:
    """Boxes per frame. With `length`, frames 1..length without boxes are included as 0."""
    if len(arr):
        f, c = np.unique(arr[:, F].astype(int), return_counts=True)
        s = pd.Series(c, index=f, name="density")
    else:
        s = pd.Series(dtype=int, name="density")
    if length:
        s = s.reindex(sorted(set(range(1, int(length) + 1)) | set(s.index)), fill_value=0)
    return s


def _bin(values, bins=DEFAULT_BINS) -> pd.Categorical:
    labels = [f"{bins[i]}-{bins[i + 1] - 1}" if bins[i + 1] < 10_000 else f">={bins[i]}"
              for i in range(len(bins) - 1)]
    return pd.cut(values, bins=list(bins), labels=labels, right=False)


def classify_switches(cm: ClearResult, trk: np.ndarray, birth_tol: int = 2) -> pd.DataFrame:
    """Label each GT identity switch as 'fragment', 'swap' or 'takeover'.

    fragment: the new track was born at the switch (object got a fresh ID)
    swap:     the previous track continues on another GT object (identities exchanged)
    takeover: anything else (previous track lost/ended, new track pre-existing)
    """
    sw = cm.switches.copy()
    if len(sw) == 0:
        sw["type"] = []
        return sw
    start = pd.Series(trk[:, F], index=trk[:, ID].astype(int)).groupby(level=0).min()
    m = cm.matches
    types = []
    for r in sw.itertuples(index=False):
        born = start.get(int(r.new_trk), -1)
        if born >= r.frame - birth_tol:
            types.append("fragment")
            continue
        cont = m[(m.trk_id == r.prev_trk) & (m.frame >= r.frame) & (m.frame <= r.frame + 5)]
        types.append("swap" if (len(cont) and (cont.gt_id != r.gt_id).any()) else "takeover")
    sw["type"] = types
    return sw


def switch_coverage(switches: pd.DataFrame, windows: pd.DataFrame, slack: int = 3) -> pd.Series:
    """'ran' / 'seed_fail' / 'none' for each switch (see module docstring)."""
    if len(switches) == 0:
        return pd.Series(dtype=object)
    if len(windows) == 0:
        return pd.Series("none", index=switches.index)
    ran = windows[windows.outcome != "SEED_FAIL"]
    failed = windows[windows.outcome == "SEED_FAIL"]
    out = []
    for r in switches.itertuples(index=False):
        ids = [r.prev_trk, r.new_trk]
        hit = ran[ran.track_id.isin(ids) & (ran.seed_frame - slack <= r.frame) & (ran.end_frame + slack >= r.frame)]
        if len(hit):
            out.append("ran")
            continue
        hit = failed[failed.track_id.isin(ids) & ((failed.entry_frame - r.frame).abs() <= slack)]
        out.append("seed_fail" if len(hit) else "none")
    return pd.Series(out, index=switches.index)


def trigger_coverage(switches: pd.DataFrame, log: pd.DataFrame, tau: float, gap_G: int,
                     slack: int = 5) -> pd.Series:
    """True if the SMP trigger rule fires on prev_trk/new_trk within ± slack frames."""
    if len(switches) == 0:
        return pd.Series(dtype=bool)
    fired = log[(log.margin_det < tau) | (log.gap >= gap_G)][["frame", "track_id"]]
    out = []
    for r in switches.itertuples(index=False):
        hit = fired[(fired.track_id.isin([r.prev_trk, r.new_trk])) & (fired.frame.between(r.frame - slack, r.frame + slack))]
        out.append(len(hit) > 0)
    return pd.Series(out, index=switches.index)


def assignment_at_switch(switches: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    """How the tracker assigned new_trk at the switch frame.

    margin_new       SMP margin of that assignment
    new_stage        ByteTrack stage: 1-2 = against all active/lost tracks;
                     3 = among unconfirmed (just-born) tracks only, where there is usually
                     a single candidate and the margin defaults to 1.0 (not informative)
    new_competitors  other tracks of that stage overlapping the detection (IoU > 0)
    All NaN when new_trk was not matched in that frame.
    """
    cols = ["margin_new", "new_stage", "new_competitors"]
    if len(switches) == 0:
        return pd.DataFrame(columns=cols, index=switches.index, dtype=float)
    lg = log.drop_duplicates(["frame", "track_id"]).set_index(["frame", "track_id"])
    rows = []
    for r in switches.itertuples(index=False):
        k = (int(r.frame), int(r.new_trk))
        if k not in lg.index:
            rows.append((np.nan, np.nan, np.nan))
            continue
        x = lg.loc[k]
        rows.append((float(x["margin_det"]),
                     float(x["stage"]) if "stage" in x else np.nan,
                     float(x["n_track_candidates"]) - 1 if "n_track_candidates" in x else np.nan))
    return pd.DataFrame(rows, columns=cols, index=switches.index)


def switch_fixed(base: pd.DataFrame, after: pd.DataFrame, tol: int = 5) -> tuple[pd.Series, int]:
    """Compare switch events before and after SMP correction (same GT matcher).

    fixed[i] = 1 if the corrected tracks have no switch on the same GT object within
    ± tol frames of base switch i. `introduced` = corrected-track switches with no base
    switch on the same GT object within ± tol frames (errors added by wrong SWAPs).
    """
    def near(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
        if len(a) == 0:
            return np.zeros(0, bool)
        if len(b) == 0:
            return np.zeros(len(a), bool)
        by = {g: d.frame.to_numpy() for g, d in b.groupby("gt_id")}
        return np.array([g in by and bool((np.abs(by[g] - f) <= tol).any())
                         for g, f in zip(a.gt_id.to_numpy(), a.frame.to_numpy())])

    fixed = pd.Series((~near(base, after)).astype(int), index=base.index)
    introduced = int((~near(after, base)).sum())
    return fixed, introduced


def pilot_tables(seq: str, gt: np.ndarray, trk: np.ndarray, log: pd.DataFrame, windows: pd.DataFrame,
                 cm: ClearResult, cm_after: Optional[ClearResult] = None, length: Optional[int] = None,
                 density: str = "gt", taus=(0.01, 0.025, 0.05), gap_G: int = 5, slack: int = 3) -> dict:
    """Per-sequence tables; concatenate across sequences with `summarise`."""
    if density not in DENSITY_LABEL:
        raise ValueError(f"density must be one of {list(DENSITY_LABEL)}")
    dens = density_per_frame(gt if density == "gt" else trk, length)
    out: dict = {}
    # H1a: share of matched pairs below each margin threshold, by density
    if len(log) == 0 or "frame" not in log:  # no association at all (e.g. a sequence without detections)
        log = pd.DataFrame(columns=["frame", "track_id", "stage", "margin_det", "gap"])
    lg = log[log.stage.isin([1, 2])].copy() if "stage" in log else log.copy()
    lg["density"] = lg.frame.map(dens).fillna(0)
    for t in taus:
        lg[f"low_margin_{t}"] = (lg.margin_det < t).astype(int)
    lg["seq"] = seq
    out["pairs"] = lg[["seq", "frame", "track_id", "density", "margin_det", "gap"] + [f"low_margin_{t}" for t in taus]]
    # H1b/P4: windows and outcomes by density at the entry frame
    fr = pd.DataFrame({"frame": dens.index, "density": dens.values})
    if len(windows):
        w = windows.copy()
        w["density"] = w.entry_frame.map(dens).fillna(0)
        w["kind"] = np.where(w.witness_of.notna(), "witness", "trigger")
        w["ran"] = (w.outcome != "SEED_FAIL").astype(int)
        w["seed_fail"] = 1 - w["ran"]
        applied = w["applied"].astype(bool) if "applied" in w else (w.outcome == "SWAP")
        w["applied"] = applied.astype(int)
        w["wasted"] = (w["ran"].astype(bool) & ~applied).astype(int)
        w["is_witness"] = (w["kind"] == "witness").astype(int)
        w["seq"] = seq
        out["windows"] = w
        for kind in ("trigger", "witness"):
            opened = w[w.kind == kind].groupby("entry_frame").size()
            fr[f"opened_{kind}"] = fr.frame.map(opened).fillna(0)
        fr["vos_frames"] = fr.frame.map(w.groupby("entry_frame").frames.sum()).fillna(0)
    else:
        out["windows"] = pd.DataFrame()
        fr["opened_trigger"] = fr["opened_witness"] = fr["vos_frames"] = 0.0
    fr["windows_opened"] = fr.opened_trigger + fr.opened_witness
    fr["seq"] = seq
    out["frames"] = fr
    # H2: switch coverage, trigger coverage, margin at the switch, fixed or not
    sw = classify_switches(cm, trk)
    out["n_introduced"] = 0
    if len(sw):
        sw["density"] = sw.frame.map(dens).fillna(0)
        cov = switch_coverage(sw, windows, slack)
        sw["covered_by_window"] = (cov == "ran").astype(int)
        sw["seed_fail_near"] = (cov == "seed_fail").astype(int)
        sw["no_window"] = (cov == "none").astype(int)
        for t in taus:
            sw[f"trigger_{t}"] = trigger_coverage(sw, log, t, gap_G).astype(int)
        sw = sw.join(assignment_at_switch(sw, log))
        if cm_after is not None:
            sw["fixed"], out["n_introduced"] = switch_fixed(sw, cm_after.switches)
        sw["seq"] = seq
    elif cm_after is not None:
        out["n_introduced"] = len(cm_after.switches)
    out["switches"] = sw
    return out


def _window_stats(w: pd.DataFrame, keys) -> pd.DataFrame:
    g = w.groupby(keys, observed=True)
    ran = g["ran"].sum()
    den = ran.where(ran > 0)
    return pd.DataFrame({
        "n_windows": g.size(),
        "witness_share": g["is_witness"].mean(),
        "seed_fail_share": g["seed_fail"].mean(),
        "n_ran": ran,
        "wasted_share_ran": g["wasted"].sum() / den,
        "mean_frames_ran": g["frames"].sum() / den,
        "vos_frames": g["frames"].sum(),
    })


def _margin_table(missed: pd.DataFrame) -> pd.DataFrame:
    """Margin of the new_trk assignment for switches with no window, by switch type."""
    rows = {}
    groups = list(missed.groupby("type")) + ([("all", missed)] if len(missed) else [])
    for name, d in groups:
        valid = d[d.new_stage.isin([1, 2])]
        rows[name] = dict(
            n_no_window=len(d),
            n_not_matched=int(d.new_stage.isna().sum()),
            n_stage3=int((d.new_stage == 3).sum()),
            n_margin_valid=len(valid),
            margin_q25=valid.margin_new.quantile(0.25) if len(valid) else np.nan,
            margin_median=valid.margin_new.median() if len(valid) else np.nan,
            margin_q75=valid.margin_new.quantile(0.75) if len(valid) else np.nan,
            no_competitor_share=(valid.new_competitors <= 0).mean() if len(valid) else np.nan,
        )
    t = pd.DataFrame.from_dict(rows, orient="index")
    t.index.name = "type"
    return t


def summarise(tables: list[dict], taus=(0.01, 0.025, 0.05), bins=DEFAULT_BINS,
              density: str = "gt") -> dict:
    cat = {k: pd.concat([t[k] for t in tables if len(t[k])], ignore_index=True) if any(len(t[k]) for t in tables)
           else pd.DataFrame() for k in ("pairs", "windows", "frames", "switches")}
    res: dict = {"_density": density}
    p = cat["pairs"]
    if len(p):
        p["bin"] = _bin(p.density, bins)
        res["h1_low_margin_share"] = p.groupby("bin", observed=True)[[f"low_margin_{t}" for t in taus]].mean().assign(
            n_pairs=p.groupby("bin", observed=True).size())
    w = cat["windows"]
    if len(w):
        w["bin"] = _bin(w.density, bins)
        res["h1_windows"] = _window_stats(w, "bin")
        res["h1_windows_by_kind"] = _window_stats(w, ["bin", "kind"]).drop(columns="witness_share")
        res["outcomes"] = pd.crosstab(w.outcome, w.kind, margins=True, margins_name="all")
    fr = cat["frames"]
    if len(fr):
        fr["bin"] = _bin(fr.density, bins)
        g = fr.groupby("bin", observed=True)
        objects = g["density"].sum()
        objects = objects.where(objects > 0)
        res["h1_windows_per_frame"] = pd.DataFrame({
            "frames": g.size(),
            "windows_per_frame": g["windows_opened"].mean(),
            "trigger_windows_per_frame": g["opened_trigger"].mean(),
            "witness_windows_per_frame": g["opened_witness"].mean(),
            "windows_per_100_objects": 100 * g["windows_opened"].sum() / objects,
            "trigger_windows_per_100_objects": 100 * g["opened_trigger"].sum() / objects,
            "vos_frames_per_frame": g["vos_frames"].mean(),
        })
    s = cat["switches"]
    if len(s):
        s["bin"] = _bin(s.density, bins)
        cols = ["covered_by_window", "seed_fail_near", "no_window"] + [f"trigger_{t}" for t in taus]
        if "fixed" in s:
            cols.append("fixed")
        res["h2_by_type"] = s.groupby("type")[cols].mean().assign(n=s.groupby("type").size())
        res["h2_by_density"] = s.groupby("bin", observed=True)[cols].mean().assign(
            n=s.groupby("bin", observed=True).size())
        res["h2_margin_of_missed"] = _margin_table(s[s.no_window == 1])
    res["_raw"] = cat
    return res


def plot_summary(res: dict, out_dir: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#dddcd8"
    s1, s2, s3 = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 (fixed order)
    xlabel = DENSITY_LABEL.get(res.get("_density", "gt"), "objects per frame")

    def new_fig():
        fig, ax = plt.subplots(figsize=(6.6, 3.8), facecolor=surface)
        ax.set_facecolor(surface)
        return fig, ax

    def style(ax, title, xl, yl):
        ax.set_title(title, loc="left", fontsize=11, color=ink)
        ax.set_xlabel(xl, color=muted)
        ax.set_ylabel(yl, color=muted)
        ax.grid(axis="y", color=grid, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)

    def save(fig, name):
        fig.tight_layout()
        p = os.path.join(out_dir, name)
        fig.savefig(p, dpi=150, facecolor=surface)
        plt.close(fig)
        paths.append(p)

    if "h1_windows" in res:
        t = res["h1_windows"]
        fig, ax = new_fig()
        x = np.arange(len(t))
        ax.plot(x, t.wasted_share_ran, marker="o", markersize=5, linewidth=2, color=s1,
                label="ran SAM but changed no ID (all windows)")
        k = res.get("h1_windows_by_kind")
        if k is not None and "trigger" in k.index.get_level_values("kind"):
            tr = k.xs("trigger", level="kind").reindex(t.index)
            ax.plot(x, tr.wasted_share_ran, marker="^", markersize=5, linewidth=2, color=s3,
                    label="same, trigger windows only")
        ax.plot(x, t.seed_fail_share, marker="s", markersize=5, linewidth=2, color=s2,
                label="seed search failed (no SAM run)")
        ax.set_xticks(x, [f"{b}\nn={n}" for b, n in zip(t.index.astype(str), t.n_windows)])
        ax.set_ylim(0, 1.02)
        style(ax, "H1 / P4: SAM windows by scene density", xlabel, "share of windows")
        ax.legend(frameon=False, fontsize=8, labelcolor=ink)
        save(fig, "h1_windows_by_density.png")
    if "h1_windows_per_frame" in res:
        t = res["h1_windows_per_frame"]
        fig, ax = new_fig()
        x = np.arange(len(t))
        ax.bar(x, t.trigger_windows_per_frame, color=s1, width=0.6, edgecolor=surface, linewidth=1.5,
               label="trigger windows")
        ax.bar(x, t.witness_windows_per_frame, bottom=t.trigger_windows_per_frame, color=s2, width=0.6,
               edgecolor=surface, linewidth=1.5, label="witness windows")
        ax.set_xticks(x, t.index.astype(str))
        style(ax, "H1: SAM windows opened per frame", xlabel, "windows / frame")
        ax.legend(frameon=False, fontsize=8, labelcolor=ink)
        save(fig, "h1_windows_per_frame.png")
    if "h2_by_type" in res:
        t = res["h2_by_type"]
        fig, ax = new_fig()
        x = np.arange(len(t))
        bottom = np.zeros(len(t))
        for col, color, label in (("covered_by_window", s1, "SAM ran over the switch"),
                                  ("seed_fail_near", s2, "window opened, seed search failed"),
                                  ("no_window", s3, "no window")):
            ax.bar(x, t[col], bottom=bottom, color=color, width=0.6, edgecolor=surface, linewidth=1.5, label=label)
            bottom += t[col].to_numpy()
        for i, n in enumerate(t.n):
            ax.text(i, 1.03, f"n={n}", ha="center", fontsize=8, color=muted)
        ax.set_xticks(x, t.index.astype(str))
        ax.set_ylim(0, 1.12)
        style(ax, "H2: real identity switches by window coverage", "switch type", "share of switches")
        ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        save(fig, "h2_coverage_by_type.png")
        stale = os.path.join(out_dir, "h2_missed_by_type.png")  # name used by older versions
        if os.path.exists(stale):
            os.remove(stale)
    return paths


def write_report(res: dict, out_dir: str, note: str = "") -> str:
    os.makedirs(out_dir, exist_ok=True)
    lines = ["# Pilot H1/H2", ""]
    if note:
        lines += [note, ""]
    for k, v in res.items():
        if k.startswith("_"):
            continue
        try:
            table = v.round(3).to_markdown()
        except ImportError:  # tabulate not installed
            table = "```\n" + v.round(3).to_string() + "\n```"
        lines += [f"## {k}", "", table, ""]
        v.to_csv(os.path.join(out_dir, f"{k}.csv"))
    path = os.path.join(out_dir, "pilot_report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path
