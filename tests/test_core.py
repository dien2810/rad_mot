"""Fast checks of the CPU pipeline on synthetic data (run: python -m pytest -q)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rad.eval.idsw import clear_match, evaluate, idf1  # noqa: E402
from rad.sam.backends import OracleBackend  # noqa: E402
from rad.sam.smp import SMPParams, apply_swaps, run_smp, smp_triggers  # noqa: E402
from rad.synth.simulate import SynthConfig, simulate  # noqa: E402
from rad.tracker.bytetrack import gt_as_detections, run_bytetrack  # noqa: E402


def _crossing(swap_at=63, n=100):
    gt, trk, log = [], [], []
    for f in range(1, n + 1):
        xa, xb = 100 + 4 * f, 600 - 4 * f
        gt += [[f, 1, xa, 300, 40, 40, 1, 3, 1], [f, 2, xb, 300, 40, 40, 1, 3, 1]]
        ida, idb = (10, 20) if f < swap_at else (20, 10)
        trk += [[f, ida, xa, 300, 40, 40, 1, 3, 1], [f, idb, xb, 300, 40, 40, 1, 3, 1]]
        m = 0.9 if abs(xa - xb) > 60 else 0.01
        log += [dict(frame=f, track_id=ida, margin_det=m, gap=1), dict(frame=f, track_id=idb, margin_det=m, gap=1)]
    return np.array(gt, float), np.array(trk, float), pd.DataFrame(log)


def test_clear_counts_swap():
    gt, trk, _ = _crossing()
    res = clear_match(gt, trk)
    assert res.num_switches == 2 and res.num_fp == 0 and res.num_fn == 0
    # CLEAR keeps the old pairing while the boxes still overlap, so the switch is
    # counted once they separate (a few frames after the tracker's swap)
    assert res.switches.frame.between(63, 67).all()


def test_idf1_perfect_and_swapped():
    gt, trk, _ = _crossing()
    assert idf1(gt, gt)["idf1"] == pytest.approx(1.0)
    assert idf1(gt, trk)["idf1"] == pytest.approx(0.64)


def test_smp_fixes_constructed_swap():
    gt, trk, log = _crossing()
    p = SMPParams(image_size=(1280, 720))
    corrected, wins, stats = run_smp(trk, log, OracleBackend(gt), smp_triggers(log, p), 100, p)
    assert stats.get("n_SWAP", 0) >= 1
    after = evaluate(gt, corrected)
    assert after["idsw"] == 0 and after["idf1"] == pytest.approx(1.0)


def test_apply_swaps_dedups_symmetric_windows():
    from rad.sam.smp import Window
    _, trk, _ = _crossing()
    w1 = Window(10, 55, "margin", outcome="SWAP", settled_id=20, swap_frame=63)
    w2 = Window(20, 55, "margin", outcome="SWAP", settled_id=10, swap_frame=63)
    out = apply_swaps(trk, [w1, w2])
    is_a = out[:, 2] == 100 + 4 * out[:, 0]   # rows of the object moving right
    assert set(out[is_a, 1].astype(int)) == {10}
    assert set(out[~is_a, 1].astype(int)) == {20}


def test_bytetrack_easy_scene_no_switches():
    cfg = SynthConfig(n_frames=120, spawn_rate=0.004, seed=1, lane_change_prob=0.0)
    gt, _ = simulate(cfg)
    det = gt_as_detections(gt, vis_min=0.0)
    trk, log = run_bytetrack(det, cfg.n_frames)
    ev = evaluate(gt, trk)
    assert ev["idsw"] == 0
    assert ev["mota"] > 0.9
    assert {"margin_det", "gap", "local_density"} <= set(log.columns)


def test_pipeline_runs_on_synthetic(tmp_path):
    from rad.dispatch.candidates import CandidateParams
    from rad.io.mot import discover_sequences
    from rad.pipeline import aggregate, load_detections, prepare, run_pilot, sweep, train_risk
    from rad.synth.simulate import write_sequence
    from rad.tracker.bytetrack import ByteTrackParams

    for i, r in enumerate([0.015, 0.03, 0.045]):
        write_sequence(str(tmp_path), f"S{i}", SynthConfig(n_frames=150, seed=i, spawn_rate=r))
    seqs = discover_sequences(str(tmp_path))
    data = [prepare(s, load_detections(s, "det"), ByteTrackParams(), CandidateParams()) for s in seqs]
    res = run_pilot(data, "oracle", SMPParams(), str(tmp_path / "out"))
    assert "per_sequence" in res and len(res["per_sequence"]) == 3
    metrics = train_risk(data, "y_risk")
    assert "roc_auc" in metrics or "error" in metrics
    df = sweep(data, "oracle", SMPParams(), "p_y_risk", taus=(0.05,), budgets=(1.0,))
    agg = aggregate(df)
    assert {"smp", "learned", "random", "bytetrack"} <= set(agg.policy)


def test_visibility_estimate_and_fasttracker_layout(tmp_path):
    from rad.io.mot import discover_sequences, write_mot_txt
    from rad.occlusion import estimate_visibility, needs_visibility
    from rad.pipeline import load_detections, load_gt

    gt, _ = simulate(SynthConfig(n_frames=60, seed=4, spawn_rate=0.03))
    flat = gt.copy()
    flat[:, 8] = 1.0
    assert needs_visibility(flat)
    est = estimate_visibility(flat, 1280, 720)
    assert np.allclose(est[:, 8], gt[:, 8])           # same depth rule as the simulator
    write_mot_txt(str(tmp_path / "FT" / "gt" / "gt.txt"), flat, tracker_format=False)  # GT only
    seq = discover_sequences(str(tmp_path))[0]
    assert seq.length == 60 and seq.width is None
    g, gv = load_gt(seq)
    det = load_detections(seq, "oracle", oracle_kw={"vis_min": 0.2}, gt_vis=gv)
    assert len(det) < len(g)                           # fully hidden objects are not detected
