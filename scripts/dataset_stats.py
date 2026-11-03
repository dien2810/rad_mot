#!/usr/bin/env python
"""Class distribution and density of a MOT-style split (first thing to run on FastTracker).

Class names follow FastTracker's GT codes (rad/classes.py; 8 = motorbike, 9 = ignore_region).
Per-sequence density and box statistics exclude the ignore classes of the config and the
GT objects lying inside those regions; class totals list every class, regions included.

Example:
    python scripts/dataset_stats.py --data datasets/FastTracker/train --out outputs/stats
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rad.classes import class_name  # noqa: E402
from rad.config import load_config  # noqa: E402
from rad.ignore import clean_gt  # noqa: E402
from rad.io.mot import C, F, H, ID, V, discover_sequences  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="split folder containing one folder per sequence")
    ap.add_argument("--out", default="outputs/stats")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    seqs = discover_sequences(args.data)
    if not seqs:
        sys.exit(f"No sequences found in {args.data}")
    os.makedirs(args.out, exist_ok=True)
    per_seq, per_class = [], []
    for s in seqs:
        raw = s.load_gt()
        if len(raw) == 0:
            continue
        cls, cnt = np.unique(raw[:, C].astype(int), return_counts=True)
        for c, n in zip(cls, cnt):
            per_class.append(dict(seq=s.name, class_id=int(c), boxes=int(n)))
        gt, regions, n_inside = clean_gt(raw, cfg["gt_ignore_classes"], cfg["det_ignore_ioa"])
        if len(gt) == 0:
            continue
        frames = np.unique(gt[:, F]).size
        dens = pd.Series(gt[:, F]).value_counts()
        per_seq.append(dict(seq=s.name, frames=frames, length=s.length, width=s.width, height=s.height,
                            fps=s.fps, boxes=len(gt), ids=np.unique(gt[:, ID]).size,
                            objects_per_frame_mean=dens.mean(), objects_per_frame_p90=dens.quantile(0.9),
                            median_box_h=float(np.median(gt[:, H])), share_box_h_lt_32=float((gt[:, H] < 32).mean()),
                            mean_visibility=float(gt[:, V].mean()), has_images=bool(s.img_dir),
                            has_video=bool(s.video_path), ignore_region_boxes=len(regions),
                            objects_inside_ignore=n_inside))
    ps = pd.DataFrame(per_seq).set_index("seq")
    pc = pd.DataFrame(per_class)
    tot = pc.groupby("class_id").boxes.sum().to_frame()
    is_obj = ~tot.index.isin([int(c) for c in cfg["gt_ignore_classes"]])
    tot["share_of_objects"] = (tot.boxes / tot.boxes[is_obj].sum()).where(is_obj)  # regions: NaN
    tot["name"] = [class_name(i) for i in tot.index]
    ps.to_csv(os.path.join(args.out, "per_sequence.csv"))
    pc.pivot(index="seq", columns="class_id", values="boxes").fillna(0).astype(int).to_csv(
        os.path.join(args.out, "class_counts_per_sequence.csv"))
    tot.to_csv(os.path.join(args.out, "class_totals.csv"))
    pd.set_option("display.width", 200)
    print("== Per sequence\n", ps.round(2).to_string())
    print(f"\n== Class totals (FastTracker GT codes; ignored as regions: {cfg['gt_ignore_classes']})\n",
          tot.round(3).to_string())
    if (ps.mean_visibility == 1.0).all():
        print("\nNote: the visibility column is constant (1.0). rad/occlusion.py estimates it from box"
              " geometry for oracle detections / oracle VOS.")
    print(f"\nSaved CSVs to {args.out}")


if __name__ == "__main__":
    main()
