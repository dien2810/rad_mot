"""scripts/spot_check_switches.py on a hand-made sequence with one real tracker error
(a 40-frame gap -> fragment) and one annotation error (two GT ids exchanged -> swap)."""
import importlib.util
import os
import subprocess
import sys
import zipfile

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "spot_check_switches.py")
sys.path.insert(0, ROOT)

spec = importlib.util.spec_from_file_location("spot_check_switches", SCRIPT)
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)

N_FRAMES, W_IMG, H_IMG = 90, 640, 360


def _objects():
    """(frame, gt_id, x, y, w, h, lane) rows; lane = physical object used to draw frames."""
    rows = []
    for f in range(1, N_FRAMES + 1):
        if f <= 20 or f >= 61:  # object A disappears for 40 frames (> track_buffer)
            rows.append((f, 1, 40 + 2 * f, 40, 60, 40, "A"))
        b_id, c_id = (2, 3) if f < 50 else (3, 2)  # annotation error: ids exchanged at frame 50
        rows.append((f, b_id, 30 + 3 * f, 170, 70, 45, "B"))
        rows.append((f, c_id, 560 - 3 * f, 280, 70, 45, "C"))
    return rows


def _make_split(tmp_path, with_frames=True):
    root = tmp_path / "train" / "SEQ"
    (root / "gt").mkdir(parents=True)
    rows = _objects()
    with open(root / "gt" / "gt.txt", "w") as fh:
        for f, i, x, y, w, h, _ in rows:
            fh.write(f"{f},{i},{x},{y},{w},{h},1,8,1.0\n")
    with open(root / "seqinfo.ini", "w") as fh:
        fh.write(f"[Sequence]\nname=SEQ\nimDir=img1\nframeRate=30\nseqLength={N_FRAMES}\n"
                 f"imWidth={W_IMG}\nimHeight={H_IMG}\nimExt=.jpg\n")
    colours = {"A": (60, 200, 60), "B": (60, 60, 220), "C": (220, 160, 40)}
    imgs = {}
    for f in range(1, N_FRAMES + 1):
        img = np.full((H_IMG, W_IMG, 3), 80, np.uint8)
        for g, _, x, y, w, h, lane in rows:
            if g == f:
                cv2.rectangle(img, (x, y), (x + w, y + h), colours[lane], -1)
        imgs[f] = img
    if with_frames:
        (root / "img1").mkdir()
        for f, img in imgs.items():
            cv2.imwrite(str(root / "img1" / f"{f:06d}.jpg"), img)
    return tmp_path / "train", imgs


def test_allocate_and_wilson():
    a = sc.allocate({"fragment": 70, "swap": 15, "takeover": 16}, 30)
    assert sum(a.values()) == 30 and a == {"fragment": 21, "swap": 4, "takeover": 5}
    assert sc.allocate({"x": 2}, 30) == {"x": 2}
    p, lo, hi = sc.wilson(0, 10)
    assert p == 0 and lo == 0 and 0.25 < hi < 0.31
    p, lo, hi = sc.wilson(5, 10)
    assert abs(p - 0.5) < 1e-9 and 0.23 < lo < 0.25 and 0.75 < hi < 0.77


def test_zip_frames(tmp_path):
    _, imgs = _make_split(tmp_path, with_frames=False)
    zp = tmp_path / "SEQ.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        for f in (3, 1, 2):  # members out of order on purpose
            ok, buf = cv2.imencode(".png", imgs[f])
            zf.writestr(f"SEQ/img1/{f:06d}.png", buf.tobytes())
        zf.writestr("SEQ/gt/gt.txt", "")
    src = sc.ZipFrames(str(zp))
    assert np.array_equal(src.get(2), imgs[2]) and src.get(4) is None
    src.close()


def test_end_to_end(tmp_path):
    data, _ = _make_split(tmp_path)
    out = tmp_path / "spot"
    r = subprocess.run([sys.executable, SCRIPT, "--data", str(data), "--out", str(out), "--n", "5"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    allsw = pd.read_csv(out / "switches_all.csv")
    assert {"fragment", "swap"} <= set(allsw.type)
    lab = pd.read_csv(out / "labels.csv", dtype={"event_id": str})
    # the two swap switches at frame 50 are one incident: only one is sampled
    assert (lab.type == "swap").sum() == 1 and (lab.type == "fragment").sum() == 1
    frag = lab[lab.type == "fragment"].iloc[0]
    swap = lab[lab.type == "swap"].iloc[0]
    assert frag.gt_gap_before > 30          # GT object absent for 40 frames
    assert swap.gt_iou_min < 0.05           # GT box jumps to the other vehicle
    assert int(swap["frame"]) == 50
    for p in list(lab["clip"]) + list(lab["sheet"]):
        assert (out / p).exists() and (out / p).stat().st_size > 0
    cap = cv2.VideoCapture(str(out / lab["clip"].iloc[0]))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n >= 50  # about +-1 s at 30 fps
    page = (out / "index.html").read_text(encoding="utf-8")
    assert "Tải labels.csv" in page and lab["sheet"].iloc[0] in page
    # label and summarise
    lab["label"] = np.where(lab.type == "swap", "gt_id", "tracker")
    lab.to_csv(out / "labels.csv", index=False)
    r = subprocess.run([sys.executable, SCRIPT, "--summarize", str(out / "labels.csv")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1/2 = 50%" in r.stdout and (out / "summary.md").exists()
    assert f"Tổng số lỗi đổi ID trong các video được lấy mẫu: {len(allsw)}" in r.stdout


def test_sequences_without_frames_are_left_out(tmp_path):
    data, _ = _make_split(tmp_path, with_frames=False)
    r = subprocess.run([sys.executable, SCRIPT, "--data", str(data), "--out", str(tmp_path / "o")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "--remote" in (r.stdout + r.stderr)


def test_compare_sweeps(tmp_path):
    spec2 = importlib.util.spec_from_file_location("compare_sweeps", os.path.join(ROOT, "scripts", "compare_sweeps.py"))
    cs = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(cs)
    a = pd.DataFrame([
        dict(seq="S1", policy="bytetrack", param=0.0, windows=0, vos_frames=0, idsw=10, idf1=0.8),
        dict(seq="S1", policy="smp", param=0.05, windows=10, vos_frames=100, idsw=9, idf1=0.81),
        dict(seq="S1", policy="learned", param=1.0, windows=5, vos_frames=50, idsw=8, idf1=0.82),
        dict(seq="S2", policy="smp", param=0.05, windows=4, vos_frames=40, idsw=5, idf1=0.9),
        dict(seq="S2", policy="learned", param=1.0, windows=2, vos_frames=20, idsw=5, idf1=0.9),
    ])
    b = a[a.seq == "S1"].copy()
    b["vos_frames"] = b["vos_frames"] * 1.2
    a.to_csv(tmp_path / "a.csv", index=False)
    (tmp_path / "b").mkdir()
    b.to_csv(tmp_path / "b" / "sweep_per_sequence.partial.csv", index=False)  # a run that died after S1
    res = cs.compare(cs.read_sweep([str(tmp_path / "a.csv")]), cs.read_sweep([str(tmp_path / "b")]))
    assert set(res["rows"].seq) == {"S1"}
    assert abs(res["ratio"]["oracle"] - 0.5) < 1e-9 and abs(res["ratio"]["sam3"] - 0.5) < 1e-9
    assert res["idsw"]["sam3"] == (9, 8)
    assert "RAD/SMP VOS frames" in cs.report(res, "oracle", "sam3")
