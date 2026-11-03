#!/usr/bin/env python
"""Extract video frames to <seq>/img1/000001.jpg ... (needed by SAM 3 and YOLO).

Only needed if a sequence ships as a video file instead of an image folder.
Example:
    python scripts/extract_frames.py --data datasets/FastTracker/train
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rad.io.mot import discover_sequences  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()
    import cv2

    for s in discover_sequences(args.data):
        if s.img_dir and s.frame_paths():
            print(f"{s.name}: frames already present ({s.img_dir})")
            continue
        if not s.video_path:
            print(f"{s.name}: no video found, skipped")
            continue
        out = os.path.join(s.root, "img1")
        os.makedirs(out, exist_ok=True)
        cap = cv2.VideoCapture(s.video_path)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            cv2.imwrite(os.path.join(out, f"{i:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        cap.release()
        print(f"{s.name}: wrote {i} frames to {out}")


if __name__ == "__main__":
    main()
