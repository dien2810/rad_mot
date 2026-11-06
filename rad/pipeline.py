"""End-to-end steps shared by the CLI scripts and the notebook.

Per sequence:  detections -> ByteTrack (+log) -> candidates/labels -> SMP windows
Across sequences: pilot H1/H2 summary, out-of-fold risk model, policy sweep.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from rad.dispatch import policy as pol
from rad.dispatch.candidates import CandidateParams, build_candidates
from rad.dispatch.labels import label_fix, label_risk
from rad.dispatch.model import train_oof
from rad.eval.idsw import clear_match, evaluate
from rad.ignore import clean_gt, drop_in_regions, split_ignore
from rad.io.mot import F, Sequence, load_mot_txt, write_mot_txt
from rad.occlusion import estimate_visibility, needs_visibility
from rad.pilot.analysis import pilot_tables, summarise
from rad.sam.backends import OracleBackend
from rad.sam.smp import SMPParams, run_smp, windows_frame
from rad.tracker.bytetrack import ByteTrackParams, gt_as_detections, run_bytetrack


@dataclass
class SeqData:
    name: str
    gt: np.ndarray
    det: np.ndarray
    length: int
    size: tuple
    tracks: Optional[np.ndarray] = None
    log: Optional[pd.DataFrame] = None
    cands: Optional[pd.DataFrame] = None
    cm: object = None
    seq: Optional[Sequence] = None
    gt_vis: Optional[np.ndarray] = None  # GT with (estimated) visibility, for oracle components


def load_gt(seq: Sequence, ignore_classes=(), ignore_ioa: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """(gt for evaluation, gt with visibility for oracle detections/VOS).

    Rows of `ignore_classes` (FastTracker: 9 = ignore_region) are regions, not objects:
    they are removed, together with GT objects lying mostly inside a region (rad/ignore.py).
    If the visibility column is constant (FastTracker), it is estimated from box geometry.
    """
    gt, _, _ = clean_gt(seq.load_gt(), ignore_classes, ignore_ioa)
    gt_vis = estimate_visibility(gt, seq.width, seq.height) if needs_visibility(gt) else gt
    return gt, gt_vis


def load_detections(seq: Sequence, source: str, det_dir: Optional[str] = None, oracle_kw: Optional[dict] = None,
                    gt_vis: Optional[np.ndarray] = None, ignore_classes=(), ignore_ioa: float = 0.5) -> np.ndarray:
    """source: 'oracle' (from GT), 'det' (seq/det/det.txt) or 'dir' (det_dir/<seq>.txt).

    Detector outputs lying mostly inside an ignore region of the GT are dropped, so the
    tracker never sees them (oracle detections come from the cleaned GT already).
    """
    if source == "oracle":
        if gt_vis is None:
            gt_vis = load_gt(seq, ignore_classes, ignore_ioa)[1]
        return gt_as_detections(gt_vis, **(oracle_kw or {}))
    if source == "det":
        if not seq.det_path:
            raise FileNotFoundError(f"{seq.name}: no det/det.txt")
        det = load_mot_txt(seq.det_path)
    elif source == "dir":
        det = load_mot_txt(os.path.join(det_dir, f"{seq.name}.txt"))
    else:
        raise ValueError(source)
    if ignore_classes and seq.gt_path:
        _, regions = split_ignore(seq.load_gt(), ignore_classes)
        det, n = drop_in_regions(det, regions, ignore_ioa)
        if n:
            print(f"{seq.name}: dropped {n} detections inside ignore regions")
    return det


def prepare(seq: Sequence, det: np.ndarray, bt: ByteTrackParams, cp: CandidateParams,
            out_dir: Optional[str] = None, ignore_classes=(), ignore_ioa: float = 0.5) -> SeqData:
    gt, gt_vis = load_gt(seq, ignore_classes, ignore_ioa)
    length = int(seq.length or max(gt[:, F].max() if len(gt) else 0, det[:, F].max() if len(det) else 0))
    size = (seq.width or int(np.ceil((gt[:, 2] + gt[:, 4]).max())) if len(gt) else 1920,
            seq.height or int(np.ceil((gt[:, 3] + gt[:, 5]).max())) if len(gt) else 1080)
    bt = ByteTrackParams(**{**asdict(bt), "frame_rate": seq.fps})
    tracks, log = run_bytetrack(det, length, bt)
    cands = build_candidates(log, tracks, cp)
    cands["seq"] = seq.name
    cm = clear_match(gt, tracks)
    cands["y_risk"] = label_risk(cands, cm)
    sd = SeqData(seq.name, gt, det, length, size, tracks, log, cands, cm, seq, gt_vis)
    if out_dir:
        d = os.path.join(out_dir, seq.name)
        os.makedirs(d, exist_ok=True)
        write_mot_txt(os.path.join(d, "bytetrack.txt"), tracks)
        log.to_csv(os.path.join(d, "assoc_log.csv.gz"), index=False)
        cands.to_csv(os.path.join(d, "candidates.csv.gz"), index=False)
    return sd


def make_backend(kind: str, sd: SeqData, **kw):
    if kind == "oracle":
        return OracleBackend(sd.gt_vis if sd.gt_vis is not None else sd.gt, **kw)
    if kind == "sam3":
        from rad.sam.backends import Sam3Backend

        return Sam3Backend(sd.seq, **kw)
    raise ValueError(kind)


def run_pilot(data: list[SeqData], backend: str, smp: SMPParams, out_dir: str, backend_kw: Optional[dict] = None,
              density: str = "gt") -> dict:
    """SMP baseline on every sequence + H1/H2 tables.

    density: 'gt' (GT objects per frame, identical across runs) or 'tracker' (tracker boxes).
    """
    tables, rows = [], []
    for sd in data:
        p = SMPParams(**{**asdict(smp), "image_size": sd.size})
        be = make_backend(backend, sd, **(backend_kw or {}))
        trig = pol.smp_baseline(sd.log, p.tau_entry, p.gap_G)
        corrected, wins, stats = run_smp(sd.tracks, sd.log, be, trig, sd.length, p)
        wdf = windows_frame(wins)
        cm_after = clear_match(sd.gt, corrected)
        tab = pilot_tables(sd.name, sd.gt, sd.tracks, sd.log, wdf, sd.cm, cm_after=cm_after, length=sd.length,
                           density=density, gap_G=p.gap_G)
        tables.append(tab)
        base, after = evaluate(sd.gt, sd.tracks), evaluate(sd.gt, corrected)
        sw = tab["switches"]
        n = max(sd.length, 1)
        rows.append(dict(seq=sd.name, objects_per_frame=len(sd.gt) / n, tracker_boxes_per_frame=len(sd.tracks) / n,
                         base_idsw=base["idsw"], base_idf1=base["idf1"], smp_idsw=after["idsw"],
                         smp_idf1=after["idf1"],
                         switches_fixed=int(sw["fixed"].sum()) if len(sw) and "fixed" in sw else 0,
                         switches_introduced=int(tab["n_introduced"]), **stats))
        d = os.path.join(out_dir, sd.name)
        os.makedirs(d, exist_ok=True)
        wdf.to_csv(os.path.join(d, f"smp_windows_{backend}.csv"), index=False)
        if len(sw):
            sw.to_csv(os.path.join(d, f"switches_{backend}.csv"), index=False)
        write_mot_txt(os.path.join(d, f"smp_{backend}.txt"), corrected)
    res = summarise(tables, density=density)
    res["per_sequence"] = pd.DataFrame(rows).set_index("seq")
    return res


def add_fix_labels(data: list[SeqData], backend: str, smp: SMPParams, backend_kw: Optional[dict] = None) -> None:
    for sd in data:
        p = SMPParams(**{**asdict(smp), "image_size": sd.size})
        be = make_backend(backend, sd, **(backend_kw or {}))
        lab = label_fix(sd.cands, sd.gt, sd.tracks, sd.log, be, sd.length, p)
        for col in lab.columns:
            sd.cands[col] = lab[col]


def train_risk(data: list[SeqData], label: str = "y_risk", kind: str = "hgb") -> dict:
    allc = pd.concat([sd.cands for sd in data], ignore_index=True)
    oof, metrics = train_oof(allc, label, kind=kind)
    allc["p_" + label] = oof
    start = 0
    for sd in data:
        n = len(sd.cands)
        sd.cands["p_" + label] = oof.iloc[start:start + n].to_numpy()
        start += n
    return metrics


def sweep(data: list[SeqData], backend: str, smp: SMPParams, score_col: str,
          taus=(0.01, 0.025, 0.05, 0.1), budgets=(0.25, 0.5, 1.0, 1.5, 2.0),
          backend_kw: Optional[dict] = None, seeds=(0,), only: Optional[set] = None,
          seqs: Optional[set] = None, partial_path: Optional[str] = None) -> pd.DataFrame:
    """Compare policies on cost (windows, VOS frames) vs accuracy (IDSW, IDF1).

    Budgets are multiples of the number of SMP(tau=0.05) triggers of each sequence, so
    learned / random policies are compared at matched trigger counts.
    only: {(policy, param)} to run, e.g. {("smp", 0.05), ("learned", 1.0)} (SAM 3 runs).
    seqs: names of the sequences to sweep (the risk model may still be trained on all).
    With the SAM 3 backend, sequences without img1/ frames are skipped.
    partial_path: CSV rewritten after every sequence (keeps results if the session dies).
    """
    rows = []
    for sd in data:
        if seqs and sd.name not in seqs:
            continue
        if backend == "sam3" and (sd.seq is None or not sd.seq.frame_paths()):
            print(f"[sweep] skip {sd.name}: no img1/ frames for SAM 3")
            continue
        p = SMPParams(**{**asdict(smp), "image_size": sd.size})
        base = evaluate(sd.gt, sd.tracks)
        rows.append(dict(seq=sd.name, policy="bytetrack", param=0.0, windows=0, vos_frames=0, triggers=0,
                         idsw=base["idsw"], idf1=base["idf1"], mota=base["mota"]))
        ref_n = len(pol.smp_baseline(sd.log, 0.05, p.gap_G))
        plans: list[tuple[str, float, int, Callable[[], pd.DataFrame]]] = []
        for t in taus:
            plans.append(("smp", t, 0, lambda t=t: pol.smp_baseline(sd.log, t, p.gap_G)))
            plans.append(("smp+birth", t, 0, lambda t=t: pol.smp_plus_births(sd.log, sd.cands, t, p.gap_G)))
            plans.append(("density", t, 0, lambda t=t: pol.density_rule(sd.cands, t, 2, p.gap_G)))
        if score_col in sd.cands and sd.cands[score_col].notna().any():
            for b in budgets:
                n = int(round(b * max(ref_n, 1)))
                plans.append(("learned", b, 0,
                              lambda n=n: pol.budget(sd.cands.dropna(subset=[score_col]), score_col, n)))
                for s in seeds:
                    plans.append(("random", b, s, lambda n=n, s=s: pol.random_budget(sd.cands, n, s)))
        if only is not None:
            plans = [x for x in plans if (x[0], round(float(x[1]), 4)) in only]
        be = make_backend(backend, sd, **(backend_kw or {}))  # one per sequence (SAM 3 model is cached)
        if backend == "sam3":  # policies often open the same window: run SAM 3 once (<= ~1 GB of masks)
            from rad.sam.record import CachingBackend

            be = CachingBackend(be, max_frames=4000)
        for name, param, seed_used, make in plans:
            trig = make()
            corrected, wins, stats = run_smp(sd.tracks, sd.log, be, trig, sd.length, p)
            ev = evaluate(sd.gt, corrected)
            rows.append(dict(seq=sd.name, policy=name, param=float(param), seed=seed_used, windows=stats["windows"],
                             vos_frames=stats["vos_frames"], triggers=len(trig), idsw=ev["idsw"],
                             idf1=ev["idf1"], mota=ev["mota"], n_swap=stats.get("n_SWAP", 0)))
        if partial_path:
            os.makedirs(os.path.dirname(os.path.abspath(partial_path)), exist_ok=True)
            pd.DataFrame(rows).to_csv(partial_path, index=False)
    return pd.DataFrame(rows)


def aggregate(sweep_df: pd.DataFrame) -> pd.DataFrame:
    """Average random seeds within each sequence, then sum counts over sequences."""
    num = ["windows", "vos_frames", "triggers", "idsw", "idf1", "mota"] + (["n_swap"] if "n_swap" in sweep_df else [])
    per_seq = sweep_df.groupby(["seq", "policy", "param"])[num].mean().reset_index()
    g = per_seq.groupby(["policy", "param"])
    out = g[[c for c in num if c not in ("idf1", "mota")]].sum()
    out["idf1_mean"] = g["idf1"].mean()
    out["mota_mean"] = g["mota"].mean()
    return out.reset_index().sort_values(["policy", "param"])


def plot_tradeoff(agg: pd.DataFrame, path: str, y: str = "idsw") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"smp": "#2f6fbd", "smp+birth": "#3a9a6b", "density": "#8a63d2", "learned": "#d4762c",
              "random": "#8c959f"}
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    base = agg[agg.policy == "bytetrack"]
    if len(base):
        ax.axhline(float(base[y].iloc[0]), color="#57606a", linewidth=0.8, linestyle="--", label="ByteTrack (no SAM)")
    for name, grp in agg[agg.policy != "bytetrack"].groupby("policy"):
        grp = grp.sort_values("vos_frames")
        ax.plot(grp.vos_frames, grp[y], marker="o", markersize=4, linewidth=1.6,
                color=colors.get(name, "#1f2328"), label=name)
    ax.set_xlabel("VOS frames processed (proxy for SAM compute)", color="#57606a")
    ax.set_ylabel({"idsw": "identity switches (lower is better)", "idf1_mean": "IDF1"}.get(y, y), color="#57606a")
    ax.set_title("Cost vs accuracy of dispatch policies", loc="left", fontsize=11)
    ax.grid(color="#d0d7de", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)
