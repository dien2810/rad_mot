"""Ignore regions (FastTracker class 9) and class codes (run: python -m pytest -q)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rad.classes import COCO_TO_FASTTRACKER, IGNORE_REGION, MOTORBIKE, class_name  # noqa: E402
from rad.ignore import clean_gt, drop_in_regions, split_ignore  # noqa: E402

REGION = [1, 900, 0, 0, 100, 100, 1, IGNORE_REGION, 1]       # frame 1, box (0,0)-(100,100)


def _gt():
    return np.array([
        REGION,
        [1, 1, 10, 10, 20, 20, 1, MOTORBIKE, 1],    # fully inside the region -> removed
        [1, 2, 80, 80, 40, 40, 1, MOTORBIKE, 1],    # 400/1600 = 25 % inside -> kept
        [2, 1, 10, 10, 20, 20, 1, MOTORBIKE, 1],    # same place, frame 2 has no region -> kept
    ], float)


def test_class_codes_follow_fasttracker_gt():
    assert class_name(8) == "motorbike" and class_name(9) == "ignore_region" and class_name(14) == "pm"
    assert COCO_TO_FASTTRACKER[3] == MOTORBIKE == 8


def test_clean_gt_removes_regions_and_objects_inside():
    objs, regions, n_inside = clean_gt(_gt(), [IGNORE_REGION])
    assert len(regions) == 1 and n_inside == 1
    assert sorted(map(tuple, objs[:, :2].astype(int))) == [(1, 2), (2, 1)]


def test_detections_inside_regions_are_dropped_per_frame():
    _, regions = split_ignore(_gt(), [IGNORE_REGION])
    det = np.array([[1, -1, 5, 5, 30, 30, 0.9, MOTORBIKE, -1],
                    [1, -1, 300, 300, 30, 30, 0.9, MOTORBIKE, -1],
                    [2, -1, 5, 5, 30, 30, 0.9, MOTORBIKE, -1]], float)
    kept, n = drop_in_regions(det, regions)
    assert n == 1 and kept[:, 0].tolist() == [1, 2] and kept[0, 2] == 300


def test_load_detections_applies_ignore_regions(tmp_path):
    from rad.io.mot import discover_sequences, write_mot_txt
    from rad.pipeline import load_detections, load_gt

    write_mot_txt(str(tmp_path / "S" / "gt" / "gt.txt"), _gt(), tracker_format=False)
    det = np.array([[1, -1, 5, 5, 30, 30, 0.9, MOTORBIKE, -1], [2, -1, 5, 5, 30, 30, 0.9, MOTORBIKE, -1]], float)
    os.makedirs(tmp_path / "dets")
    write_mot_txt(str(tmp_path / "dets" / "S.txt"), det, tracker_format=False)
    seq = discover_sequences(str(tmp_path))[0]
    assert len(load_detections(seq, "dir", str(tmp_path / "dets"))) == 2                 # no ignore classes
    assert len(load_detections(seq, "dir", str(tmp_path / "dets"), ignore_classes=[9])) == 1
    gt, _ = load_gt(seq, [9])
    assert len(gt) == 2 and not (gt[:, 7] == IGNORE_REGION).any()
