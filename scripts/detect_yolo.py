#!/usr/bin/env python
"""Run an Ultralytics YOLO detector over every sequence and write <out>/<seq>.txt (MOT det format).

COCO classes are mapped to FastTracker's GT codes (rad/classes.py: person 1, bike 7, car 6,
motorbike 8; bus -> 2 bus_small and truck -> 4 truck_small, since COCO has no size split),
so per-class analysis (e.g. --classes 8 = motorbikes) matches the GT. The raw file keeps
every detection; detections inside ignore regions are dropped when they are loaded.
Fine-tuning the detector on the FastTracker GT gives better detections; pretrained COCO
weights are enough for the pilot.

Example (GPU recommended):
    python scripts/detect_yolo.py --data datasets/FastTracker/train --weights yolo11x.pt --out outputs/dets
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rad.classes import COCO_TO_FASTTRACKER  # noqa: E402
from rad.io.mot import discover_sequences, write_mot_txt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default="yolo11x.pt")
    ap.add_argument("--out", default="outputs/dets")
    ap.add_argument("--conf", type=float, default=0.05, help="keep low scores for ByteTrack's 2nd stage")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--half", action="store_true", help="fp16 inference (faster on T4)")
    ap.add_argument("--seqs", nargs="*", default=None, help="only these sequences")
    ap.add_argument("--skip-existing", action="store_true", help="skip sequences whose <out>/<seq>.txt exists")
    args = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    os.makedirs(args.out, exist_ok=True)
    for s in discover_sequences(args.data):
        if args.seqs and s.name not in args.seqs:
            continue
        out_path = os.path.join(args.out, f"{s.name}.txt")
        if args.skip_existing and os.path.exists(out_path):
            print(f"{s.name}: {out_path} exists, skipped")
            continue
        paths = s.frame_paths()
        if not paths:
            print(f"{s.name}: no frames (run scripts/extract_frames.py first)")
            continue
        rows = []
        for start in range(0, len(paths), args.batch):
            chunk = paths[start:start + args.batch]
            results = model.predict(chunk, conf=args.conf, imgsz=args.imgsz, device=args.device,
                                    classes=list(COCO_TO_FASTTRACKER), half=args.half, verbose=False)
            for k, r in enumerate(results):
                frame = start + k + 1
                b = r.boxes
                if b is None or len(b) == 0:
                    continue
                xyxy = b.xyxy.cpu().numpy()
                conf = b.conf.cpu().numpy()
                cls = b.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), c, k_cls in zip(xyxy, conf, cls):
                    rows.append([frame, -1, x1, y1, x2 - x1, y2 - y1, c, COCO_TO_FASTTRACKER.get(int(k_cls), -1), -1])
        arr = np.asarray(rows, dtype=float) if rows else np.zeros((0, 9))
        tmp = out_path + ".part"
        write_mot_txt(tmp, arr, tracker_format=False)
        os.replace(tmp, out_path)  # a run killed mid-sequence leaves no half-written file
        print(f"{s.name}: {len(arr)} detections over {len(paths)} frames")


if __name__ == "__main__":
    main()
