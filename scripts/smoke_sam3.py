#!/usr/bin/env python
"""Week-1 check: can SAM 3 run on this GPU? Measures time per frame and peak VRAM.

Prompts SAM 3 with GT boxes of the N largest objects of one frame (one window per
object, like SMP) and reports cost. Needs: CUDA GPU, `pip install -e sam3`, HF login
with access to facebook/sam3, and image frames in <seq>/img1.

Example:
    python scripts/smoke_sam3.py --data datasets/FastTracker/train --seq <name> --frame 100 --objects 5 --len 60
    # on a T4, also try: --precision fp16  (compare ms/frame and the "mask stays on its GT box" counts)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rad.io.mot import F, ID, X, Y, W, H, discover_sequences  # noqa: E402
from rad.sam.backends import Sam3Backend  # noqa: E402
from rad.sam.masks import ioma  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq", default=None)
    ap.add_argument("--frame", type=int, default=50)
    ap.add_argument("--objects", type=int, default=5)
    ap.add_argument("--len", type=int, default=60)
    ap.add_argument("--precision", choices=["auto", "fp16"], default="auto",
                    help="auto = SAM 3 default (bf16; emulated on T4). fp16 = experimental for T4")
    args = ap.parse_args()
    print("GPU:", Sam3Backend.gpu_info())
    seqs = discover_sequences(args.data)
    if args.seq:
        seq = next((s for s in seqs if s.name == args.seq), None)
        if seq is None:
            sys.exit(f"no sequence named {args.seq} in {args.data}")
    else:  # default: the first sequence whose frames were downloaded (--mode gt has no img1/)
        seq = next((s for s in seqs if s.frame_paths()), None)
        if seq is None:
            sys.exit("no sequence has img1/ frames yet: run scripts/download_fasttracker.py --mode full --seqs <name>")
    print("sequence:", seq.name)
    gt = seq.load_gt()
    rows = gt[gt[:, F] == args.frame]
    rows = rows[np.argsort(-(rows[:, W] * rows[:, H]))][: args.objects]
    be = Sam3Backend(seq, precision=args.precision)
    for r in rows:
        box = np.array([r[X], r[Y], r[X] + r[W], r[Y] + r[H]])
        gid = int(r[ID])
        hits = 0
        n = 0
        for f, mask, score in be.propagate(args.frame, box, args.len):
            g = gt[(gt[:, F] == f) & (gt[:, ID] == gid)]
            if len(g):
                gb = np.array([[g[0, X], g[0, Y], g[0, X] + g[0, W], g[0, Y] + g[0, H]]])
                hits += int(ioma(mask, gb)[0] >= 0.5)
                n += 1
        print(f"object {gid}: mask stays on its GT box in {hits}/{n} frames")
    st = be.stats()
    print(f"\n{seq.name} [{args.precision}]: {st['calls']} windows, {st['frames']} frames, "
          f"{st['sec_per_frame'] * 1000:.0f} ms/frame, peak VRAM {st['peak_mem_gb']:.1f} GB")
    print("Note: the first window includes model warm-up; run with more --objects for a stable number.")
    print(f"Use this with: python scripts/estimate_gpu_time.py --ms-per-frame {st['sec_per_frame'] * 1000:.0f} ...")


if __name__ == "__main__":
    main()
