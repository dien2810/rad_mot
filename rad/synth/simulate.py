"""Synthetic dense two-wheeler traffic (no images) for testing the pipeline on CPU.

Produces MOT-style GT (with visibility from depth-ordered occlusion) and noisy
detections. Objects weave between lanes (non-linear motion) and overtake each
other, which creates the overlaps and identity ambiguities studied in this project.

NOTE: synthetic results are only for checking that the code works. They are not
evidence about real traffic.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from rad.classes import CAR, MOTORBIKE  # FastTracker GT codes (8, 6)
from rad.io.mot import write_mot_txt


@dataclass
class SynthConfig:
    width: int = 1280
    height: int = 720
    n_frames: int = 400
    fps: float = 30.0
    n_lanes: int = 6
    lane_top: float = 220.0
    lane_bottom: float = 690.0
    spawn_rate: float = 0.025         # expected spawns per lane per frame (~40 objects/frame)
    p_motorbike: float = 0.8
    mb_size: tuple = (34.0, 46.0)     # (w, h) at the bottom of the image
    car_size: tuple = (110.0, 72.0)
    speed_mb: tuple = (4.0, 9.0)      # px/frame at the bottom of the image
    speed_car: tuple = (4.0, 7.0)
    lane_change_prob: float = 0.012   # per object per frame
    lateral_jitter: float = 0.5
    # detector simulation
    det_vis_min: float = 0.2
    det_miss_base: float = 0.02
    det_jitter: float = 0.04
    fp_per_frame: float = 0.4
    cls_confusion: float = 0.03
    seed: int = 0


@dataclass
class _Obj:
    oid: int
    cls: int
    x: float          # left edge at reference scale position
    yb: float         # bottom y
    speed: float
    target_yb: float
    t_start: int = 0
    t_len: int = 0
    y_from: float = 0.0
    alive: bool = True
    path: list = field(default_factory=list)


def _scale(cfg: SynthConfig, yb: float) -> float:
    t = (yb - cfg.lane_top) / max(cfg.lane_bottom - cfg.lane_top, 1.0)
    return float(np.clip(0.45 + 0.55 * t, 0.35, 1.05))


def _visibility(boxes: np.ndarray, depth: np.ndarray, w: int, h: int, cell: int = 4) -> np.ndarray:
    """Fraction of each box not covered by boxes closer to the camera (larger depth)."""
    gw, gh = int(np.ceil(w / cell)), int(np.ceil(h / cell))
    owner = -np.ones((gh, gw), dtype=int)
    order = np.argsort(depth)  # far first, near last overwrite
    for k in order:
        x1, y1, x2, y2 = boxes[k]
        c1, r1 = max(int(x1 // cell), 0), max(int(y1 // cell), 0)
        c2, r2 = min(int(np.ceil(x2 / cell)), gw), min(int(np.ceil(y2 / cell)), gh)
        if c2 > c1 and r2 > r1:
            owner[r1:r2, c1:c2] = k
    vis = np.zeros(len(boxes))
    for k in range(len(boxes)):
        x1, y1, x2, y2 = boxes[k]
        c1, r1 = max(int(x1 // cell), 0), max(int(y1 // cell), 0)
        c2, r2 = min(int(np.ceil(x2 / cell)), gw), min(int(np.ceil(y2 / cell)), gh)
        full_w = max(np.ceil(x2 / cell) - x1 // cell, 1)
        full_h = max(np.ceil(y2 / cell) - y1 // cell, 1)
        if c2 > c1 and r2 > r1:
            vis[k] = (owner[r1:r2, c1:c2] == k).sum() / (full_w * full_h)
    return np.clip(vis, 0.0, 1.0)


def simulate(cfg: SynthConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (gt, det) arrays in the internal 9-column layout."""
    rng = np.random.default_rng(cfg.seed)
    lanes = np.linspace(cfg.lane_top + 20, cfg.lane_bottom - 10, cfg.n_lanes)
    lane_gap = lanes[1] - lanes[0] if cfg.n_lanes > 1 else 60.0
    objs: list[_Obj] = []
    next_id = 1
    gt_rows, det_rows = [], []

    # warm-up so the first frame is already dense
    warm = int(cfg.width / np.mean(cfg.speed_mb)) + 10
    for t in range(-warm, cfg.n_frames):
        frame = t + 1
        # spawn
        for lane_y in lanes:
            for _ in range(rng.poisson(cfg.spawn_rate)):
                is_mb = rng.random() < cfg.p_motorbike
                cls = MOTORBIKE if is_mb else CAR
                sp = rng.uniform(*(cfg.speed_mb if is_mb else cfg.speed_car))
                yb = lane_y + rng.normal(0, 4)
                objs.append(_Obj(next_id, cls, -rng.uniform(40, 140), yb, sp, yb))
                next_id += 1
        # move
        for o in objs:
            if not o.alive:
                continue
            s = _scale(cfg, o.yb)
            o.speed = float(np.clip(o.speed + rng.normal(0, 0.15), 2.0, 11.0))
            o.x += o.speed * s
            if o.t_len == 0 and o.cls == MOTORBIKE and rng.random() < cfg.lane_change_prob:
                direction = rng.choice([-1, 1])
                tgt = o.yb + direction * lane_gap * rng.uniform(0.5, 1.1)
                o.target_yb = float(np.clip(tgt, cfg.lane_top + 10, cfg.lane_bottom))
                o.y_from, o.t_start, o.t_len = o.yb, t, int(rng.integers(20, 45))
            if o.t_len > 0:
                u = min((t - o.t_start) / o.t_len, 1.0)
                o.yb = o.y_from + (o.target_yb - o.y_from) * (0.5 - 0.5 * np.cos(np.pi * u))
                if u >= 1.0:
                    o.t_len = 0
            o.yb += rng.normal(0, cfg.lateral_jitter)
            if o.x > cfg.width + 10:
                o.alive = False
        objs = [o for o in objs if o.alive]
        if t < 0:
            continue
        # boxes
        boxes, meta = [], []
        for o in objs:
            s = _scale(cfg, o.yb)
            bw, bh = (cfg.mb_size if o.cls == MOTORBIKE else cfg.car_size)
            bw, bh = bw * s, bh * s
            x1, y1, x2, y2 = o.x, o.yb - bh, o.x + bw, o.yb
            cx1, cy1 = max(x1, 0.0), max(y1, 0.0)
            cx2, cy2 = min(x2, cfg.width - 1.0), min(y2, cfg.height - 1.0)
            if cx2 - cx1 < 4 or cy2 - cy1 < 4:
                continue
            boxes.append([cx1, cy1, cx2, cy2])
            meta.append(o)
        if not boxes:
            continue
        boxes = np.asarray(boxes)
        depth = boxes[:, 3]
        vis = _visibility(boxes, depth, cfg.width, cfg.height)
        for (x1, y1, x2, y2), o, v in zip(boxes, meta, vis):
            gt_rows.append([frame, o.oid, x1, y1, x2 - x1, y2 - y1, 1.0, o.cls, v])
            # detection
            if v < cfg.det_vis_min:
                continue
            if rng.random() < cfg.det_miss_base + 0.3 * (1.0 - v):
                continue
            w, h = x2 - x1, y2 - y1
            jx, jy = rng.normal(0, cfg.det_jitter * w, 2)
            jw, jh = rng.normal(0, cfg.det_jitter * w), rng.normal(0, cfg.det_jitter * h)
            if v < 0.8:  # partially occluded: box shrinks from the top (occluder in front)
                shrink = (1.0 - v) * rng.uniform(0.0, 0.4) * h
                y1 += shrink
                h -= shrink
            score = float(np.clip(0.35 + 0.6 * v + rng.normal(0, 0.05), 0.05, 0.99))
            cls = o.cls
            if rng.random() < cfg.cls_confusion:
                cls = CAR if cls == MOTORBIKE else MOTORBIKE
            det_rows.append([frame, -1, x1 + jx, y1 + jy, max(w + jw, 3), max(h + jh, 3), score, cls, -1])
        for _ in range(rng.poisson(cfg.fp_per_frame)):
            yb = rng.uniform(cfg.lane_top, cfg.lane_bottom)
            s = _scale(cfg, yb)
            w, h = cfg.mb_size[0] * s, cfg.mb_size[1] * s
            det_rows.append([frame, -1, rng.uniform(0, cfg.width - w), yb - h, w, h,
                             float(rng.uniform(0.1, 0.55)), MOTORBIKE, -1])
    gt = np.asarray(gt_rows, dtype=float) if gt_rows else np.zeros((0, 9))
    det = np.asarray(det_rows, dtype=float) if det_rows else np.zeros((0, 9))
    return gt, det


def write_sequence(out_dir: str, name: str, cfg: SynthConfig) -> str:
    """Simulate and write <out_dir>/<name>/{gt/gt.txt, det/det.txt, seqinfo.ini}."""
    gt, det = simulate(cfg)
    root = os.path.join(out_dir, name)
    write_mot_txt(os.path.join(root, "gt", "gt.txt"), gt, tracker_format=False)
    write_mot_txt(os.path.join(root, "det", "det.txt"), det, tracker_format=False)
    with open(os.path.join(root, "seqinfo.ini"), "w") as fh:
        fh.write(
            "[Sequence]\n"
            f"name={name}\nimDir=img1\nframeRate={cfg.fps}\nseqLength={cfg.n_frames}\n"
            f"imWidth={cfg.width}\nimHeight={cfg.height}\nimExt=.jpg\n"
        )
    return root
