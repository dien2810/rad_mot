"""scripts/run_sam3_dispatch.py on synthetic data with the oracle VOS (no GPU, no frames)."""
import os
import subprocess
import sys

import pandas as pd

from rad.synth.simulate import SynthConfig, write_sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "run_sam3_dispatch.py")


def _run(*args):
    r = subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_runs_resumes_and_summarises(tmp_path):
    data = tmp_path / "train"
    for i, rate in enumerate((0.02, 0.035, 0.05)):
        write_sequence(str(data), f"S{i}", SynthConfig(n_frames=160, seed=i, spawn_rate=rate))
    out = tmp_path / "out"
    log = _run("--data", str(data), "--backend", "oracle", "--run", "S2", "S0", "--out", str(out))
    assert sorted(os.listdir(out / "per_seq")) == ["S0.csv", "S2.csv"]
    assert (out / "risk_scores.csv.gz").exists()
    per = pd.read_csv(out / "per_seq" / "S2.csv")
    assert list(per.policy) == ["bytetrack", "learned", "smp"]  # RAD before SMP
    smp, rad = per.set_index("policy").loc["smp"], per.set_index("policy").loc["learned"]
    assert rad.triggers == smp.triggers or abs(rad.triggers - smp.triggers) <= 1
    assert smp.sec_candidates == 0 and smp.sec_score == 0 and rad.sec_candidates > 0
    assert (per.fps > 0).all() and (per.sec_total >= per.sec_vos).all()
    assert (per.model_frames <= per.vos_frames + 3).all()
    assert "frames_on_gpu" in per and (per.frames_on_gpu == 0).all()

    # second call: S0 and S2 are skipped, S1 runs, cached scores are reused
    log = _run("--data", str(data), "--backend", "oracle", "--out", str(out))
    assert "reused out-of-fold scores" in log and "[skip] S0" in log and "[skip] S2" in log
    assert sorted(os.listdir(out / "per_seq")) == ["S0.csv", "S1.csv", "S2.csv"]

    # summary merges an older untimed sweep for videos that are not in per_seq
    old = pd.DataFrame([dict(seq="OLD", policy=p, param=v, windows=1, vos_frames=10, triggers=1, idsw=3,
                             idf1=0.9, mota=0.9) for p, v in (("bytetrack", 0.0), ("smp", 0.05), ("learned", 1.0))])
    old.to_csv(tmp_path / "old.csv", index=False)
    _run("--out", str(out), "--summary-only", "--old", str(tmp_path / "old.csv"))
    summ = pd.read_csv(out / "summary.csv").set_index("policy")
    assert summ.loc["smp", "videos"] == 4 and summ.loc["smp", "timed_videos"] == 3
    assert summ.loc["learned", "fps"] > 0
    assert "VOS frames RAD/SMP" in (out / "summary.md").read_text()
