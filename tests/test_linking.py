"""Tracklet linking (method 3) and VOS verification (method 4)."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from rad.link import LinkParams, apply_links, assign, label_links, link_candidates, verify_pair
from rad.sam.backends import OracleBackend
from rad.synth.simulate import SynthConfig, write_sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rows(tid, frames, x0, vx=5.0, y=100.0, w=40.0, h=80.0, score=0.9):
    return [[f, tid, x0 + vx * (f - frames[0]), y, w, h, score, 6, 1.0] for f in frames]


def _fragment_scene():
    """Object 1 is tracked as A=1 (frames 1-20), lost 10 frames, restarted as C=2 (31-50).
    Object 2 is an unrelated car far away (track 3)."""
    trk = np.array(_rows(1, range(1, 21), 100) + _rows(2, range(31, 51), 100 + 5 * 30)
                   + _rows(3, range(1, 51), 1200, vx=0.0), dtype=float)
    gt = np.array(_rows(10, range(1, 51), 100) + _rows(20, range(1, 51), 1200, vx=0.0), dtype=float)
    return trk, gt


def test_candidates_labels_and_merge():
    trk, gt = _fragment_scene()
    lc = link_candidates(trk, (1920, 1080), 30.0, LinkParams())
    assert set(zip(lc.a, lc.c)) == {(1, 2)}  # track 3 never ends before 2 starts
    pair = lc.iloc[0]
    assert pair["dt"] == 11 and pair["dist_ext"] < 0.5  # extrapolated position lands on C
    from rad.eval.idsw import clear_match, evaluate
    lc["y"] = label_links(lc, clear_match(gt, trk).matches, 10)
    assert lc.y.tolist() == [1]
    merged = apply_links(trk, assign(lc.assign(p=0.9), "p", 0.5))
    assert set(np.unique(merged[:, 1]).astype(int)) == {1, 3}
    assert evaluate(gt, merged)["idsw"] < evaluate(gt, trk)["idsw"]


def test_assign_is_one_to_one_and_chains_relabel():
    lc = pd.DataFrame(dict(a=[1, 1, 4, 2], c=[2, 5, 5, 6], p=[0.9, 0.8, 0.7, 0.95]))
    sel = assign(lc, "p", 0.5)
    assert sel.a.is_unique and sel.c.is_unique and len(sel) == 3  # (1,2), (4,5), (2,6)
    trk = np.array([[f, i, 0, 0, 1, 1, 1, 6, 1] for i, f in [(1, 1), (2, 5), (6, 9), (4, 1), (5, 6)]], float)
    out = apply_links(trk, sel)
    assert out[:, 1].tolist() == [1, 1, 1, 4, 4]  # 6 -> 2 -> 1


def test_verify_pair_confirms_true_link_and_rejects_wrong_one():
    trk, gt = _fragment_scene()
    be = OracleBackend(gt, vis_min=0.0)
    ok = verify_pair(be, trk, 1, 2, LinkParams())
    assert ok["confirmed"] and ok["ioma_c"] > 0.9
    wrong = trk.copy()
    wrong[wrong[:, 1] == 2, 2] += 600  # C now far from where the object really is
    bad = verify_pair(be, wrong, 1, 2, LinkParams())
    assert not bad["confirmed"]


def test_script_end_to_end(tmp_path):
    data = tmp_path / "train"
    for i, rate in enumerate((0.02, 0.035, 0.05)):
        write_sequence(str(data), f"S{i}", SynthConfig(n_frames=200, seed=i, spawn_rate=rate))
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_linking.py"), "--data", str(data),
                        "--dets", "det", "--backend", "oracle", "--max-verify", "5", "--out", str(out)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(os.listdir(out / "per_seq")) == ["S0.csv", "S1.csv", "S2.csv"]
    per = pd.read_csv(out / "per_seq" / "S2.csv").set_index("policy")
    assert list(per.index) == ["bytetrack", "link_upper", "link", "link_verify"]
    assert per.loc["link_verify", "verify_calls"] <= 5
    assert per.loc["link_upper", "idsw"] <= per.loc["bytetrack", "idsw"]
    assert (out / "summary.md").exists() and (out / "model_metrics.json").exists()
