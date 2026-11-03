#!/usr/bin/env python
"""Write a synthetic dense-traffic split (GT + noisy detections, no images) for testing.

Example:
    python scripts/make_synthetic.py --out datasets/SYNTH/train --n 6
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rad.synth.simulate import SynthConfig, write_sequence  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6, help="number of sequences (density increases with index)")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--min-rate", type=float, default=0.012)
    ap.add_argument("--max-rate", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=10)
    args = ap.parse_args()
    for i, rate in enumerate(np.linspace(args.min_rate, args.max_rate, args.n)):
        root = write_sequence(args.out, f"SYN-{i:02d}", SynthConfig(n_frames=args.frames, seed=args.seed + i,
                                                                    spawn_rate=float(rate)))
        print("wrote", root)


if __name__ == "__main__":
    main()
