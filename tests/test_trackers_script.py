"""scripts/run_trackers.py end to end on synthetic data (ByteTrack only: OC-SORT needs its own clone)."""
import os
import subprocess
import sys

import pandas as pd

from rad.synth.simulate import SynthConfig, write_sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_bytetrack_and_link(tmp_path):
    data = tmp_path / "train"
    for i, rate in enumerate((0.02, 0.035, 0.05)):
        write_sequence(str(data), f"S{i}", SynthConfig(n_frames=200, seed=i, spawn_rate=rate))
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_trackers.py"), "--data", str(data),
                        "--dets", "det", "--trackers", "bytetrack", "--link", "--out", str(out)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    summ = pd.read_csv(out / "summary.csv", index_col=0)
    assert list(summ.index) == ["bytetrack", "bytetrack+link"]
    assert (summ.tracker_fps > 0).all() and (summ.pipeline_fps < summ.tracker_fps).all()
    assert sorted(os.listdir(out / "tracks" / "bytetrack")) == ["S0.txt", "S1.txt", "S2.txt"]
