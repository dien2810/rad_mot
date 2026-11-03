#!/usr/bin/env python
"""Risk-Aware Dispatch: labels -> out-of-fold risk model -> policy sweep (cost vs accuracy).

Example (CPU, oracle VOS):
    python scripts/run_dispatch.py --data datasets/FastTracker/train --dets oracle --backend oracle \
        --label y_risk --out outputs/dispatch
With SAM 3 (GPU): --backend sam3 (fix labels with --fix-backend oracle to keep labelling cheap).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rad.config import load_config  # noqa: E402
from rad.io.mot import discover_sequences  # noqa: E402
from rad.pipeline import (add_fix_labels, aggregate, load_detections, load_gt, plot_tradeoff,  # noqa: E402
                          prepare, save_json, sweep, train_risk)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    ap.add_argument("--dets", choices=["oracle", "det", "dir"], default="oracle")
    ap.add_argument("--det-dir", default=None)
    ap.add_argument("--backend", choices=["oracle", "sam3"], default="oracle", help="VOS used in the sweep")
    ap.add_argument("--label", choices=["y_risk", "y_fix"], default="y_risk")
    ap.add_argument("--fix-backend", choices=["oracle", "sam3"], default="oracle", help="VOS used for y_fix")
    ap.add_argument("--model", choices=["hgb", "lgbm"], default="hgb")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--out", default="outputs/dispatch")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seqs = discover_sequences(args.data)
    if args.seqs:
        seqs = [s for s in seqs if s.name in set(args.seqs)]
    data = []
    for s in seqs:
        ign = dict(ignore_classes=cfg["gt_ignore_classes"], ignore_ioa=cfg["det_ignore_ioa"])
        _, gt_vis = load_gt(s, **ign)
        det = load_detections(s, args.dets, args.det_dir, cfg["oracle_dets"], gt_vis=gt_vis, **ign)
        data.append(prepare(s, det, cfg["bytetrack"], cfg["candidates"], out_dir=args.out,
                            **ign))
    if args.label == "y_fix":
        t = time.time()
        kw = cfg["oracle_backend"] if args.fix_backend == "oracle" else cfg["sam3_backend"]
        add_fix_labels(data, args.fix_backend, cfg["smp"], backend_kw=kw)
        print(f"[labels] y_fix done ({time.time() - t:.1f}s)")
    metrics = train_risk(data, args.label, kind=args.model)
    print("[model]", {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
    save_json(metrics, os.path.join(args.out, f"model_metrics_{args.label}.json"))
    sw_cfg = cfg["sweep"]
    kw = cfg["oracle_backend"] if args.backend == "oracle" else cfg["sam3_backend"]
    t = time.time()
    df = sweep(data, args.backend, cfg["smp"], "p_" + args.label, taus=tuple(sw_cfg.get("taus", (0.05,))),
               budgets=tuple(sw_cfg.get("budgets", (1.0,))), backend_kw=kw,
               seeds=tuple(sw_cfg.get("random_seeds", (0,))))
    print(f"[sweep] {len(df)} runs ({time.time() - t:.1f}s)")
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "sweep_per_sequence.csv"), index=False)
    agg = aggregate(df)
    agg.to_csv(os.path.join(args.out, "sweep_summary.csv"), index=False)
    print(agg.round(3).to_string(index=False))
    for y in ("idsw", "idf1_mean"):
        print("plot:", plot_tradeoff(agg, os.path.join(args.out, f"tradeoff_{y}.png"), y=y))


if __name__ == "__main__":
    main()
