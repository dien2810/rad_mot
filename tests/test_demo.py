"""End-to-end check of scripts/make_demo.py on a tiny synthetic split with rendered frames."""
import json
import os
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rad.io.mot import H, ID, W, X, Y, group_by_frame, load_mot_txt  # noqa: E402
from rad.synth.simulate import SynthConfig, write_sequence  # noqa: E402


def _split(tmp_path, n_frames=90):
    out = tmp_path / "train"
    for i, rate in enumerate([0.03, 0.04]):
        cfg = SynthConfig(width=640, height=360, lane_top=110, lane_bottom=345, n_frames=n_frames,
                          seed=5 + i, spawn_rate=rate)
        root = write_sequence(str(out), f"S{i}", cfg)
        by = group_by_frame(load_mot_txt(os.path.join(root, "gt", "gt.txt")))
        os.makedirs(os.path.join(root, "img1"))
        for f in range(1, n_frames + 1):
            img = np.full((360, 640, 3), 90, np.uint8)
            for r in by.get(f, np.zeros((0, 9))):
                c = (37 * int(r[ID])) % 255
                cv2.rectangle(img, (int(r[X]), int(r[Y])), (int(r[X] + r[W]), int(r[Y] + r[H])), (c, 255 - c, 128), -1)
            cv2.imwrite(os.path.join(root, "img1", f"{f:06d}.jpg"), img)
    return out


def test_make_demo_runs_and_writes_outputs(tmp_path):
    data = _split(tmp_path)
    out = tmp_path / "demo"
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "make_demo.py"), "--data", str(data), "--seq", "S1",
           "--dets", "det", "--backend", "oracle", "--width", "640", "--out", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stdout + res.stderr
    (d,) = list(out.iterdir())
    summary = json.loads((d / "summary.json").read_text())
    assert summary["clip"][0] >= 1 and summary["clip"][1] <= 90
    assert set(summary["clip_eval"]) == {"base", "smp", "rad"}
    assert summary["triggers_whole_sequence"]["smp"] == summary["triggers_whole_sequence"]["rad"]
    frames = sorted((d / "keyframes").glob("*.png"))
    assert frames
    img = cv2.imread(str(frames[0]))
    assert img.shape == (360, 640, 3)
    assert (d / "windows_smp.csv").exists() and (d / "events.csv").exists()
    if summary["video"]:
        assert os.path.getsize(summary["video"]) > 1000


def test_make_demo_bundle_for_the_browser_page(tmp_path):
    data = _split(tmp_path)
    out = tmp_path / "demo"
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "make_demo.py"), "--data", str(data), "--seq", "S1",
           "--dets", "det", "--backend", "oracle", "--width", "640", "--no-video", "--out", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stdout + res.stderr
    (d,) = list(out.iterdir())
    b = json.loads((d / "bundle.json").read_text())
    s0, s1 = b["clip"]
    assert b["has_gt"] and set(b["boxes"]) == {"gt", "base", "smp", "rad"}
    assert all(s0 <= r[0] <= s1 for r in b["boxes"]["base"])
    assert set(b["methods"]) == {"base", "smp", "rad"} and b["methods"]["base"]["eval"]["idsw"] == len(b["switches"]["base"])
    assert all("fixed_smp" in s and "fixed_rad" in s for s in b["switches"]["base"])
    assert (d / "bundle.js").read_text().startswith("window.RAD_BUNDLE = ")
    cap = cv2.VideoCapture(str(d / "clip.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert abs(n - (s1 - s0 + 1)) <= 1


def test_make_demo_without_ground_truth(tmp_path):
    data = _split(tmp_path)
    os.remove(os.path.join(str(data), "S1", "gt", "gt.txt"))  # a new video: frames and detections only
    out = tmp_path / "demo"
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "make_demo.py"), "--data", str(data), "--seq", "S1",
           "--dets", "det", "--backend", "oracle", "--width", "640", "--frames", "60", "--out", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stdout + res.stderr
    (d,) = list(out.iterdir())
    summary = json.loads((d / "summary.json").read_text())
    assert summary["has_gt"] is False and summary["clip"] == [1, 60]
    b = json.loads((d / "bundle.json").read_text())
    assert "gt" not in b["boxes"] and b["switches"]["base"] == [] and b["methods"]["smp"]["eval"] is None


def test_demo_app_handlers(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("demo_app", os.path.join(ROOT, "scripts", "demo_app.py"))
    da = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(da)
    data = _split(tmp_path)
    app = da.App(str(data), str(tmp_path / "work"), None, None, "yolo11l.pt", "fp16")
    assert da.sequences_with_frames(str(data)) == ["S0", "S1"]
    *_, last = app.run_fasttracker("S1", False, 10, 60, "Từ nhãn GT (oracle)", "VOS oracle (CPU, nhanh)")
    log, summary, events, files, video = last
    assert "Lỗi đổi ID" in summary and "khung 10–69" in summary, log
    assert {os.path.basename(f) for f in files} >= {"bundle.json", "clip.mp4", "summary.json"}
    *_, bad = app.run_fasttracker("", True, 1, 10, "Từ nhãn GT (oracle)", "VOS oracle (CPU, nhanh)")
    assert bad[1].startswith("**Lỗi:**")
    d, n, fps = da.extract_frames(files[[os.path.basename(f) for f in files].index("clip.mp4")],
                                  str(tmp_path / "up"), "up", 0.5, 1.0)
    assert n == round(fps) and os.path.exists(os.path.join(d, "up", "seqinfo.ini"))
