#!/usr/bin/env python
"""Pilot H1/H2: ByteTrack + SMP baseline, then density/coverage analysis.

CPU-only mode (no images, no GPU): --dets oracle --backend oracle
  -> detections built from GT, VOS = GT-following oracle. Measures how the SMP trigger
     and window mechanism behave as density grows, independent of detector/SAM quality.
Realistic mode: --dets dir --det-dir outputs/dets --backend sam3 (GPU + frames).

Density (x axis of every H1/H2 table) is GT objects per frame by default, so bins are the
same for oracle, YOLO and SAM 3 runs; --density tracker uses tracker boxes instead.

Witness sensitivity (how much of the cost/benefit comes from witness windows):
    --no-witness          only windows opened by the trigger rule
    --witness-iou 0.3     witness windows only for neighbours with IoU > 0.3 (config: 0.0)

Example:
    python scripts/run_pilot.py --data datasets/FastTracker/train --dets oracle --backend oracle --out outputs/pilot
    python scripts/run_pilot.py --data datasets/FastTracker/train --no-witness --out outputs/pilot_nowit
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rad.config import load_config  # noqa: E402
from rad.io.mot import discover_sequences  # noqa: E402
from rad.pilot.analysis import plot_summary, write_report  # noqa: E402
from rad.pipeline import load_detections, load_gt, prepare, run_pilot  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    ap.add_argument("--dets", choices=["oracle", "det", "dir"], default="oracle")
    ap.add_argument("--det-dir", default=None)
    ap.add_argument("--backend", choices=["oracle", "sam3"], default="oracle")
    ap.add_argument("--seqs", nargs="*", default=None, help="subset of sequence names")
    ap.add_argument("--out", default="outputs/pilot")
    ap.add_argument("--density", choices=["gt", "tracker"], default="gt",
                    help="density used for every H1/H2 bin (default: GT objects per frame)")
    ap.add_argument("--no-witness", action="store_true", help="disable witness windows")
    ap.add_argument("--witness-iou", type=float, default=None, help="override smp.witness_iou")
    args = ap.parse_args()
    cfg = load_config(args.config)
    smp = cfg["smp"]
    if args.no_witness:
        smp = replace(smp, witness=False)
    if args.witness_iou is not None:
        smp = replace(smp, witness_iou=args.witness_iou)
    seqs = discover_sequences(args.data)
    if args.seqs:
        seqs = [s for s in seqs if s.name in set(args.seqs)]
    if not seqs:
        sys.exit("no sequences")
    data = []
    for s in seqs:
        t = time.time()
        ign = dict(ignore_classes=cfg["gt_ignore_classes"], ignore_ioa=cfg["det_ignore_ioa"])
        _, gt_vis = load_gt(s, **ign)
        det = load_detections(s, args.dets, args.det_dir, cfg["oracle_dets"], gt_vis=gt_vis, **ign)
        data.append(prepare(s, det, cfg["bytetrack"], cfg["candidates"], out_dir=args.out,
                            **ign))
        print(f"[track] {s.name}: {len(det)} dets, {len(data[-1].tracks)} track boxes ({time.time() - t:.1f}s)")
    kw = cfg["oracle_backend"] if args.backend == "oracle" else cfg["sam3_backend"]
    res = run_pilot(data, args.backend, smp, args.out, backend_kw=kw, density=args.density)
    note = (f"data={args.data} dets={args.dets} backend={args.backend} density={args.density} "
            f"witness={smp.witness} witness_iou={smp.witness_iou}. "
            "Oracle backend = GT-following VOS (upper bound for VOS quality).")
    report = write_report(res, args.out, note)
    plots = plot_summary(res, args.out)
    print(res["per_sequence"].round(3).to_string())
    for k in ("h1_windows", "h1_windows_per_frame", "h2_by_type", "h2_margin_of_missed"):
        if k in res:
            print(f"\n== {k}\n{res[k].round(3).to_string()}")
    print("\nreport:", report)
    print("plots:", *plots, sep="\n  ")


if __name__ == "__main__":
    main()
