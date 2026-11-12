#!/usr/bin/env python
"""Pure multi-object trackers on the same detections, same GT, same metrics.

Trackers (all class-agnostic, all fed the same detection file per video):
  bytetrack    - this repo's ByteTrack (the baseline of every other experiment)
  ocsort       - official OC-SORT (Cao et al., CVPR 2023), default parameters of its repo
                 (det_thresh 0.6, IoU 0.3, min_hits 3, delta_t 3, inertia 0.2, no BYTE)
  ocsort_byte  - the same with its BYTE second association (use_byte=True)
With --link, each tracker is also run with method 3 on top (learned tracklet linking, no SAM):
the link model is trained on that tracker's own tracks, out of fold by video, thresholds
chosen inside the training folds, exactly as in scripts/run_linking.py.

Metrics per video and in total: IDSW and its types (fragment / swap / takeover), IDF1, MOTA
from this repo's CLEAR-MOT, and HOTA / DetA / AssA from TrackEval when it is installed.
Speed: seconds of the tracker alone; with --det-sec-per-frame (default: YOLO11x at 1280 px on
a T4, measured 10/11/2026) also the frame rate of detector + tracker.

OC-SORT is the official code (MIT licence): clone it and pass its folder with --ocsort-dir
    git clone https://github.com/noahcao/OC_SORT /content/OC_SORT

Example:
    python scripts/run_trackers.py --data datasets/FastTracker/train --dets dir --det-dir outputs/dets_yolo \
        --ocsort-dir /content/OC_SORT --link --out outputs/trackers_yolo
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from rad.config import load_config  # noqa: E402
from rad.eval.idsw import clear_match  # noqa: E402
from rad.eval.trackeval_runner import available as trackeval_available, run_trackeval  # noqa: E402
from rad.io.mot import F, H, S, W, X, Y, discover_sequences, write_mot_txt  # noqa: E402
from rad.link import LinkParams, apply_links, assign, label_links, link_candidates  # noqa: E402
from rad.pipeline import load_detections, load_gt  # noqa: E402
from rad.tracker.bytetrack import ByteTrackParams, run_bytetrack  # noqa: E402
from run_linking import bootstrap_sum, metrics, nested_scores  # noqa: E402

YOLO11X_SEC_PER_FRAME = 1776.2 / 19480  # 12 FastTracker videos, 1280 px, fp16, T4 (from file times)


def run_ocsort(det: np.ndarray, length: int, size: tuple, fps: float, ocsort_cls, use_byte: bool) -> np.ndarray:
    """Official OC-SORT, frame by frame; max_age scaled with the frame rate like ByteTrack's buffer."""
    w, h = size
    trk = ocsort_cls(det_thresh=0.6, max_age=max(1, int(round(30 * fps / 30.0))), min_hits=3, iou_threshold=0.3,
                     delta_t=3, asso_func="iou", inertia=0.2, use_byte=use_byte)
    frames = det[:, F].astype(int)
    order = np.argsort(frames, kind="stable")
    det, frames = det[order], frames[order]
    starts = np.searchsorted(frames, np.arange(1, length + 2))
    rows = []
    for f in range(1, length + 1):
        d = det[starts[f - 1]:starts[f]]
        arr = (np.stack([d[:, X], d[:, Y], d[:, X] + d[:, W], d[:, Y] + d[:, H], d[:, S]], 1)
               if len(d) else np.empty((0, 5)))
        for x1, y1, x2, y2, tid in trk.update(arr, (h, w), (h, w)):
            rows.append([f, int(tid), x1, y1, x2 - x1, y2 - y1, 1.0, -1, -1])
    return np.asarray(rows, dtype=float) if rows else np.zeros((0, 9))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default=os.path.join(HERE, "..", "configs", "default.yaml"))
    ap.add_argument("--dets", choices=["dir", "det", "oracle"], default="dir")
    ap.add_argument("--det-dir", default="outputs/dets_yolo")
    ap.add_argument("--trackers", nargs="*", default=["bytetrack", "ocsort", "ocsort_byte"])
    ap.add_argument("--ocsort-dir", default="OC_SORT", help="folder of a clone of github.com/noahcao/OC_SORT")
    ap.add_argument("--link", action="store_true", help="also run each tracker + method 3 (learned linking)")
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--det-sec-per-frame", type=float, default=YOLO11X_SEC_PER_FRAME,
                    help="detector time per frame for the pipeline frame rate (0: tracker only)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ign = dict(ignore_classes=cfg["gt_ignore_classes"], ignore_ioa=cfg["det_ignore_ioa"])
    ocsort_cls = None
    if any(t.startswith("ocsort") for t in args.trackers):
        sys.path.insert(0, os.path.abspath(args.ocsort_dir))
        try:
            from trackers.ocsort_tracker.ocsort import OCSort as ocsort_cls  # noqa: N813
        except ImportError as e:
            sys.exit(f"OC-SORT not found in {args.ocsort_dir} ({e}); clone github.com/noahcao/OC_SORT there")
    os.makedirs(args.out, exist_ok=True)

    seqs, rows, tracks_by = [], [], {}
    for s in discover_sequences(args.data):
        gt, gt_vis = load_gt(s, **ign)
        det = load_detections(s, args.dets, args.det_dir, cfg["oracle_dets"], gt_vis=gt_vis, **ign)
        length = int(s.length or max(gt[:, F].max() if len(gt) else 0, det[:, F].max() if len(det) else 0))
        size = (s.width or 1920, s.height or 1080)
        fps = float(s.fps or 30.0)
        seqs.append(dict(name=s.name, gt=gt, length=length, size=size, fps=fps))
        for name in args.trackers:
            t = time.perf_counter()
            if name == "bytetrack":
                trk, _ = run_bytetrack(det, length, ByteTrackParams(**{**asdict(cfg["bytetrack"]), "frame_rate": fps}))
            elif name in ("ocsort", "ocsort_byte"):
                trk = run_ocsort(det, length, size, fps, ocsort_cls, use_byte=name == "ocsort_byte")
            else:
                sys.exit(f"unknown tracker {name}")
            sec = time.perf_counter() - t
            tracks_by[(name, s.name)] = trk
            write_mot_txt(os.path.join(args.out, "tracks", name, f"{s.name}.txt"), trk)
            rows.append(dict(seq=s.name, tracker=name, frames=length, sec_tracker=sec, **metrics(gt, trk)))
            print(f"[{s.name}] {name}: IDSW {rows[-1]['idsw']}, IDF1 {rows[-1]['idf1']:.3f}, "
                  f"{length / max(sec, 1e-9):.0f} frames/s", flush=True)

    if args.link:
        lp = LinkParams()
        for name in args.trackers:
            parts = []
            for sq in seqs:
                trk = tracks_by[(name, sq["name"])]
                t = time.perf_counter()
                lc = link_candidates(trk, sq["size"], sq["fps"], lp)
                lc["seq"] = sq["name"]
                lc["sec"] = time.perf_counter() - t
                lc["y"] = label_links(lc, clear_match(sq["gt"], trk).matches, lp.label_k)
                parts.append(lc)
            lc_all = pd.concat(parts, ignore_index=True)
            lc_all["p"], taus = nested_scores(lc_all, args.lam, 0.2)
            for sq in seqs:
                trk = tracks_by[(name, sq["name"])]
                lc = lc_all[lc_all.seq == sq["name"]]
                t = time.perf_counter()
                sel = assign(lc, "p", float(taus.loc[sq["name"], "tau_high"]))
                linked = apply_links(trk, sel)
                sec = time.perf_counter() - t + (float(lc["sec"].iloc[0]) if len(lc) else 0.0)
                base = next(r for r in rows if r["seq"] == sq["name"] and r["tracker"] == name)
                tracks_by[(name + "+link", sq["name"])] = linked
                write_mot_txt(os.path.join(args.out, "tracks", name + "+link", f"{sq['name']}.txt"), linked)
                rows.append(dict(seq=sq["name"], tracker=name + "+link", frames=sq["length"],
                                 sec_tracker=base["sec_tracker"] + sec, merges=len(sel),
                                 merges_wrong=int((sel.y == 0).sum()), **metrics(sq["gt"], linked)))
            print(f"[link] {name}: {int(lc_all.y.sum())} true pairs of {len(lc_all)}", flush=True)

    per = pd.DataFrame(rows)
    variants = list(dict.fromkeys(per.tracker))
    if trackeval_available():
        for v in variants:
            te = run_trackeval({sq["name"]: sq["gt"] for sq in seqs},
                               {sq["name"]: tracks_by[(v, sq["name"])] for sq in seqs},
                               {sq["name"]: sq["length"] for sq in seqs}, tracker_name=v.replace("+", "_"))
            for k in ("HOTA", "DetA", "AssA"):
                per.loc[per.tracker == v, k] = per.loc[per.tracker == v, "seq"].map(lambda s: te[s][k])
                per.loc[per.tracker == v, k + "_all"] = te["COMBINED"][k]
    else:
        print("[trackeval] not installed: HOTA skipped (pip install git+https://github.com/JonathonLuiten/TrackEval)")
    per.to_csv(os.path.join(args.out, "per_seq.csv"), index=False)

    g = per.groupby("tracker", sort=False)
    summ = g[["idsw", "fragment", "swap", "takeover"]].sum()
    summ["idf1_mean"] = g.idf1.mean()
    summ["mota_mean"] = g.mota.mean()
    for k in ("HOTA", "DetA", "AssA"):
        if k + "_all" in per:
            summ[k] = g[k + "_all"].first()
    summ["tracker_fps"] = g.frames.sum() / g.sec_tracker.sum()
    if args.det_sec_per_frame > 0:
        summ["pipeline_fps"] = g.frames.sum() / (g.sec_tracker.sum() + args.det_sec_per_frame * g.frames.sum())
    summ = summ.reindex(variants)
    summ.to_csv(os.path.join(args.out, "summary.csv"))
    lines = ["# Trackers on the same detections", "", summ.round(3).to_string(), ""]
    piv = per.pivot_table(index="seq", columns="tracker", values="idsw")
    if "bytetrack" in piv:
        for v in variants:
            if v != "bytetrack":
                d = (piv.bytetrack - piv[v]).to_numpy()
                lo, hi = bootstrap_sum(d)
                lines.append(f"IDSW reduction vs ByteTrack, {v}: {d.sum():.0f} (95% bootstrap: {lo:.0f} to {hi:.0f}); "
                             f"better on {(d > 0).sum()}, worse on {(d < 0).sum()} of {len(d)} videos")
    txt = "\n".join(lines)
    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
