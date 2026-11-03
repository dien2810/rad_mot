#!/usr/bin/env python
"""HOTA / MOTA / IDF1 with TrackEval for a folder of tracker results (<seq>.txt).

Ignore regions follow the config (gt_ignore_classes, det_ignore_ioa; FastTracker: class 9):
region rows and GT objects inside them are removed, and tracker boxes lying mostly inside
a region are dropped before evaluation, the same rule as in run_pilot / run_dispatch.

Example:
    python scripts/eval_trackeval.py --data datasets/FastTracker/train --results outputs/pilot --name bytetrack.txt
    (results may be <results>/<seq>.txt or <results>/<seq>/<name>)
    --classes 8 evaluates motorbikes only (FastTracker GT codes, rad/classes.py).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rad.config import load_config  # noqa: E402
from rad.eval.trackeval_runner import run_trackeval  # noqa: E402
from rad.ignore import clean_gt, drop_in_regions  # noqa: E402
from rad.io.mot import discover_sequences, load_mot_txt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--name", default=None, help="file name inside <results>/<seq>/ (else <results>/<seq>.txt)")
    ap.add_argument("--classes", nargs="*", type=int, default=None, help="evaluate only these GT classes")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    ap.add_argument("--ignore-classes", nargs="*", type=int, default=None,
                    help="override the config's gt_ignore_classes (pass nothing to disable)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ignore = cfg["gt_ignore_classes"] if args.ignore_classes is None else args.ignore_classes
    gts, trks, lens = {}, {}, {}
    for s in discover_sequences(args.data):
        p = os.path.join(args.results, s.name, args.name) if args.name else os.path.join(args.results, f"{s.name}.txt")
        if not os.path.exists(p):
            print("missing", p)
            continue
        gt, regions, _ = clean_gt(s.load_gt(), ignore, cfg["det_ignore_ioa"])
        trk, n = drop_in_regions(load_mot_txt(p), regions, cfg["det_ignore_ioa"])
        if n:
            print(f"{s.name}: dropped {n} tracker boxes inside ignore regions")
        gts[s.name], trks[s.name], lens[s.name] = gt, trk, int(s.length)
    res = run_trackeval(gts, trks, lens, filter_classes=args.classes)
    for k, v in res.items():
        print(f"{k:>20}: " + "  ".join(f"{m}={x:.3f}" if isinstance(x, float) else f"{m}={x}" for m, x in v.items()))


if __name__ == "__main__":
    main()
