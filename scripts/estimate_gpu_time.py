#!/usr/bin/env python
"""Estimate SAM 3 GPU time before spending it.

The oracle runs (CPU) already count how many frames SAM would have to process
(`vos_frames`). Multiply by the ms/frame measured with scripts/smoke_sam3.py on your
GPU to get the hours each experiment needs.

Example:
    python scripts/estimate_gpu_time.py --ms-per-frame 250 \
        --pilot outputs/pilot/per_sequence.csv --sweep outputs/dispatch/sweep_summary.csv
"""
from __future__ import annotations

import argparse
import os

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ms-per-frame", type=float, required=True, help="from scripts/smoke_sam3.py")
    ap.add_argument("--pilot", default="outputs/pilot/per_sequence.csv")
    ap.add_argument("--sweep", default="outputs/dispatch/sweep_summary.csv")
    ap.add_argument("--session-hours", type=float, default=3.0,
                    help="GPU hours you can realistically use per Colab session")
    args = ap.parse_args()
    sec = args.ms_per_frame / 1000.0
    if os.path.exists(args.pilot):
        p = pd.read_csv(args.pilot)
        p["sam3_hours"] = p["vos_frames"] * sec / 3600
        print("== Pilot with SAM 3 (SMP baseline, per sequence)")
        print(p[["seq", "objects_per_frame", "windows", "vos_frames", "sam3_hours"]].round(2).to_string(index=False))
        tot = p["sam3_hours"].sum()
        print(f"total: {tot:.1f} h  (~{tot / args.session_hours:.1f} sessions of {args.session_hours:g} h)\n")
    if os.path.exists(args.sweep):
        s = pd.read_csv(args.sweep)
        s = s[s.policy != "bytetrack"].copy()
        s["sam3_hours"] = s["vos_frames"] * sec / 3600
        print("== Policy sweep with SAM 3 (all sequences, per policy setting)")
        print(s[["policy", "param", "windows", "vos_frames", "sam3_hours"]].round(2).to_string(index=False))
        print(f"whole sweep: {s['sam3_hours'].sum():.1f} h. Usually only a few settings need SAM 3 "
              "(e.g. smp@0.05 and learned@1.0); keep the rest on the oracle backend.")


if __name__ == "__main__":
    main()
