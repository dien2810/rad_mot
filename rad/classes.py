"""Class codes used everywhere in the project = the codes in FastTracker's gt/gt.txt.

Source: FastTracker repository (github.com/Hamidreza-Hashempoor/FastTracker), file
TrackEval/trackeval/datasets/mot_challenge_2d_box.py, `class_name_to_class_id`
(commit d35eeda, 2025-12-03). The original spells 13 as 'heaavy_equipment'.

FastTracker's class-aware tracker (yolox/tracker/fasttracker_cls.py) uses a different,
0-based 9-class index for its detector (0 person ... 8 bus_big). It is NOT used here.
"""
from __future__ import annotations

FASTTRACKER_GT_CLASSES = {
    1: "person", 2: "bus_small", 3: "bus_big", 4: "truck_small", 5: "truck_big", 6: "car", 7: "bike",
    8: "motorbike", 9: "ignore_region", 10: "tractor", 11: "trailor", 12: "wheelchair",
    13: "heavy_equipment", 14: "pm", 15: "umbrella",
}

PERSON, BUS_SMALL, TRUCK_SMALL, CAR, BIKE, MOTORBIKE, IGNORE_REGION = 1, 2, 4, 6, 7, 8, 9

# COCO (Ultralytics YOLO) class id -> FastTracker GT code. COCO has no small/big split for
# buses and trucks, so they get one representative code each (bus_small, truck_small);
# use per-class analysis only for classes that map one-to-one (motorbike, car, person, bike).
COCO_TO_FASTTRACKER = {0: PERSON, 1: BIKE, 2: CAR, 3: MOTORBIKE, 5: BUS_SMALL, 7: TRUCK_SMALL}


def class_name(code: int) -> str:
    return FASTTRACKER_GT_CLASSES.get(int(code), "?")
