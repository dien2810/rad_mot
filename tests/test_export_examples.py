"""scripts/export_switch_examples.py: ranking of switch events and rendering of one clip."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))
sys.path.insert(0, ROOT)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ex = _load("export_switch_examples", os.path.join(os.path.dirname(ROOT), "scripts", "export_switch_examples.py"))
tsc = _load("test_spot_check", os.path.join(ROOT, "test_spot_check.py"))


def _ev(**kw):
    base = dict(seq="task_day_x", frame=100, gt_id=1, prev_trk=5, new_trk=12, type="fragment", box_w=120.0,
                box_h=90.0, gt_gap_before=0, gt_iou_min=0.8, gt_vis=0.9, n_gt_overlap=1, gt_dup=1)
    base.update(kw)
    return base


def test_pick_filters_and_ranks():
    pop = pd.DataFrame([
        _ev(frame=100, box_h=90),                                   # kept, 2nd
        _ev(frame=300, gt_id=2, box_h=150),                         # kept, best
        _ev(frame=500, gt_id=3, box_h=40),                          # too small
        _ev(frame=700, gt_id=4, gt_gap_before=40),                  # hole in the labels
        _ev(frame=900, gt_id=5, gt_dup=2),                          # duplicate GT id
        _ev(frame=1100, gt_id=6, type="swap", gt_iou_min=0.05),     # GT box jumped: annotation error
        _ev(frame=1300, gt_id=7, type="swap", seq="task_night_y"),  # kept
        _ev(frame=1500, gt_id=8, type="takeover", gt_iou_min=np.nan),
    ])
    sel = ex.pick(pop, per_type=3, min_h=60)
    frag = sel[sel.type == "fragment"]
    assert list(frag.frame) == [300, 100]
    assert list(sel[sel.type == "swap"].frame) == [1300]
    assert list(sel[sel.type == "takeover"].frame) == [1500]


def test_export_renders_clip_and_poster(tmp_path):
    split, _ = tsc._make_split(tmp_path)
    cfg = ex.load_config(os.path.join(os.path.dirname(ROOT), "configs", "default.yaml"))
    seq = ex.discover_sequences(str(split))[0]
    tr = ex.sc.track_oracle(seq, cfg)
    cand = ex.candidates(seq, tr)
    assert len(cand) >= 1 and {"box_h", "gt_gap_before", "type"} <= set(cand.columns)
    ev = cand.iloc[0].to_dict()
    src, how = ex.sc.frame_source(seq, False, None)
    out = tmp_path / "clip.mp4"
    poster = tmp_path / "clip.png"
    try:
        f0, f1 = ex.export(ev, tr, src, (640, 360), seq.fps, str(out), str(poster), 0.5, 0.5, 0.5)
    finally:
        src.close()
    assert how == "img1" and f0 < int(ev["frame"]) <= f1
    assert out.stat().st_size > 1000 and poster.exists()
