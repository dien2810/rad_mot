"""Checks of the pilot H1/H2 tables (run: python -m pytest -q)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rad.eval.idsw import clear_match  # noqa: E402
from rad.pilot.analysis import (assignment_at_switch, density_per_frame, pilot_tables,  # noqa: E402
                                summarise, switch_coverage, switch_fixed)
from rad.sam.backends import OracleBackend  # noqa: E402
from rad.sam.smp import SMPParams, Window, apply_swaps, run_smp, smp_triggers, windows_frame  # noqa: E402
from test_core import _crossing  # noqa: E402


def _switch(frame=60, prev=10, new=20):
    return pd.DataFrame([dict(frame=frame, gt_id=1, prev_trk=prev, new_trk=new)])


def test_seed_fail_window_is_not_coverage():
    sw = _switch()
    failed = windows_frame([Window(10, 60, "margin", outcome="SEED_FAIL")])
    assert switch_coverage(sw, failed).tolist() == ["seed_fail"]
    ran = windows_frame([Window(20, 62, "margin", seed_frame=50, end_frame=80, outcome="CLEAN", frames=31)])
    assert switch_coverage(sw, ran).tolist() == ["ran"]
    far = windows_frame([Window(10, 90, "margin", seed_frame=85, end_frame=120, outcome="CLEAN", frames=36)])
    assert switch_coverage(sw, far).tolist() == ["none"]


def test_apply_swaps_marks_only_the_applied_window():
    _, trk, _ = _crossing()
    w1 = Window(10, 55, "margin", outcome="SWAP", settled_id=20, swap_frame=63)
    w2 = Window(20, 55, "margin", witness_of=10, outcome="SWAP", settled_id=10, swap_frame=63)
    apply_swaps(trk, [w1, w2])
    assert [w1.applied, w2.applied] == [True, False]


def test_density_counts_empty_frames():
    arr = np.array([[2, 1, 0, 0, 1, 1, 1, 1, 1], [2, 2, 0, 0, 1, 1, 1, 1, 1]], float)
    d = density_per_frame(arr, length=3)
    assert d.to_dict() == {1: 0, 2: 2, 3: 0}


def test_assignment_stage_and_competitors():
    log = pd.DataFrame([dict(frame=60, track_id=20, margin_det=1.0, stage=3, n_track_candidates=1)])
    a = assignment_at_switch(_switch(), log).iloc[0]
    assert a.new_stage == 3 and a.new_competitors == 0 and a.margin_new == 1.0


def test_fixed_and_introduced_on_constructed_swap():
    gt, trk, log = _crossing()
    p = SMPParams(image_size=(1280, 720))
    corrected, wins, stats = run_smp(trk, log, OracleBackend(gt), smp_triggers(log, p), 100, p)
    base, after = clear_match(gt, trk), clear_match(gt, corrected)
    fixed, introduced = switch_fixed(base.switches, after.switches)
    assert fixed.tolist() == [1, 1] and introduced == 0
    assert stats["swaps_applied"] == 1 and stats["windows_trigger"] + stats["windows_witness"] == stats["windows"]

    wdf = windows_frame(wins)
    tab = pilot_tables("X", gt, trk, log, wdf, base, cm_after=after, length=100)
    res = summarise([tab])
    # every window is counted once in each H1 table
    assert res["h1_windows"].n_windows.sum() == len(wdf)
    assert tab["frames"].windows_opened.sum() == len(wdf)
    assert res["outcomes"].loc["all", "all"] == len(wdf)
    # a duplicate SWAP (the second window of the A<->B pair) counts as wasted, not useful
    w = tab["windows"]
    assert w.applied.sum() == 1 and w.wasted.sum() == w.ran.sum() - 1
    assert res["h2_by_type"].fixed.mean() == 1.0


def test_pilot_tables_handles_sequence_without_detections():
    gt = np.array([[1, 1, 0, 0, 10, 10, 1, 8, 1]], float)
    trk = np.zeros((0, 9))
    tab = pilot_tables("EMPTY", gt, trk, pd.DataFrame(), pd.DataFrame(), clear_match(gt, trk), length=3)
    assert len(tab["pairs"]) == 0 and tab["frames"].windows_opened.sum() == 0
