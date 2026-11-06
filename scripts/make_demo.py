#!/usr/bin/env python
"""Demo video: ByteTrack vs SMP vs RAD (learned dispatch) on one clip, with the SAM masks.

What it does
  1. ByteTrack on the whole sequence (detections: GT oracle or a YOLO folder).
  2. Risk model trained on the OTHER sequences (leave-this-video-out), scores of this one.
  3. Triggers of both policies on the whole sequence:
       SMP  = margin < tau or re-found after a gap (rad.dispatch.policy.smp_baseline)
       RAD  = top-n candidates by predicted risk, n = budget x number of SMP triggers
  4. Picks a clip (default: around an ID switch that a policy fixes, found with the cheap
     oracle VOS), then runs only the windows of that clip with the chosen VOS backend
     (SAM 3 on GPU, or oracle on CPU). Windows shared by both policies run once (cache).
  5. Renders a 2x2 video: ByteTrack | SMP / RAD | cost and ID-switch curves, plus
     key frames (PNG), per-policy windows (CSV) and summary.json.
  6. Writes bundle.json (+ bundle.js) and clip.mp4 for the browser page demo/index.html:
     boxes of GT and of the three methods, ID switches (fixed / introduced), windows.

Without ground truth (a new video, e.g. from scripts/demo_app.py) nothing can be scored:
the clip starts at --start (default 1), the video shows windows instead of ID switches and
the bundle has no GT boxes and no switches.

Examples
  # CPU preview (oracle VOS draws boxes instead of masks), YOLO detections:
  python scripts/make_demo.py --data $DATA --seq task_day_left_turn --dets dir --det-dir $WORK/outputs/dets \
      --backend oracle --train-cands "$WORK/outputs/dispatch/*/candidates.csv.gz" --out $WORK/outputs/demo
  # the real demo on a GPU (same command, --backend sam3)
  python scripts/make_demo.py ... --backend sam3
  # a fixed clip instead of the automatic one
  python scripts/make_demo.py ... --start 380 --frames 300

Note: the corrected IDs are shown from the swap frame on (offline correction, as in SMP).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import asdict

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rad.config import load_config  # noqa: E402
from rad.dispatch import policy as pol  # noqa: E402
from rad.dispatch.candidates import FEATURES  # noqa: E402
from rad.eval.idsw import clear_match, evaluate  # noqa: E402
from rad.io.mot import F, ID, X, Y, W, H, discover_sequences  # noqa: E402
from rad.pilot.analysis import classify_switches, switch_fixed  # noqa: E402
from rad.pipeline import load_detections, load_gt, make_backend, prepare  # noqa: E402
from rad.sam.record import CachingBackend, RecordingBackend, attach_shapes  # noqa: E402
from rad.sam.smp import SMPParams, run_smp, windows_frame  # noqa: E402
from rad.viz.demo import PanelData, Text, VideoWriter, View, draw_chart, draw_panel, draw_texts  # noqa: E402

DET_NAME = {"oracle": "detection từ GT (oracle)", "det": "detection có sẵn", "dir": "detection YOLO"}
METHOD_NAME = {"base": "ByteTrack", "smp": "SMP", "rad": "RAD"}


# ----------------------------------------------------------------------------- browser bundle
def _int(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else int(v)


def _box_rows(a: np.ndarray, s0: int, s1: int) -> list:
    a = a[(a[:, F] >= s0) & (a[:, F] <= s1)]
    return [[int(r[F]), int(r[ID]), round(float(r[X]), 1), round(float(r[Y]), 1), round(float(r[W]), 1),
             round(float(r[H]), 1)] for r in a]


def _near(a: pd.DataFrame, b: pd.DataFrame, tol: int = 5) -> np.ndarray:
    """a[i] has a switch of the same GT object in b within +-tol frames."""
    if len(a) == 0:
        return np.zeros(0, bool)
    if len(b) == 0:
        return np.zeros(len(a), bool)
    by = {g: d.frame.to_numpy() for g, d in b.groupby("gt_id")}
    return np.array([g in by and bool((np.abs(by[g] - f) <= tol).any())
                     for g, f in zip(a.gt_id.to_numpy(), a.frame.to_numpy())])


def write_bundle(out_dir: str, meta: dict, gt: np.ndarray | None, tracks: dict, cms: dict, runs: dict,
                 s0: int, s1: int) -> str:
    """bundle.json / bundle.js for demo/index.html (one clip, three methods)."""
    sw = {}
    for key, cm in cms.items():
        if cm is None:
            sw[key] = pd.DataFrame(columns=["frame", "gt_id", "prev_trk", "new_trk", "type"])
        else:
            sw[key] = classify_switches(cm, tracks[key])
    base = sw["base"]
    switches = {}
    rows = []
    for k in range(len(base)):
        r = base.iloc[k]
        rows.append(dict(frame=int(r.frame), gt_id=int(r.gt_id), prev=int(r.prev_trk), new=int(r.new_trk),
                         type=str(r.type)))
    for key in ("smp", "rad"):
        still = _near(base, sw[key])
        for row, s in zip(rows, still):
            row[f"fixed_{key}"] = not bool(s)
    switches["base"] = rows
    for key in ("smp", "rad"):
        known = _near(sw[key], base)
        switches[key] = [dict(frame=int(r.frame), gt_id=int(r.gt_id), prev=int(r.prev_trk), new=int(r.new_trk),
                              type=str(r.type), introduced=not bool(kn))
                         for r, kn in zip(sw[key].itertuples(index=False), known)]
    windows = {}
    for key, run in runs.items():
        windows[key] = [dict(track=int(w.track_id), witness_of=_int(w.witness_of), trigger=w.trigger,
                             entry=int(w.entry_frame), seed=_int(w.seed_frame), end=_int(w.end_frame),
                             swap=_int(w.swap_frame), outcome=w.outcome, frames=int(w.frames),
                             applied=bool(w.applied), settled=_int(w.settled_id), score=run["scores"].get(i))
                        for i, w in enumerate(run["windows"])]
    boxes = {key: _box_rows(t, s0, s1) for key, t in tracks.items()}
    if gt is not None and len(gt):
        boxes["gt"] = _box_rows(gt, s0, s1)
    bundle = dict(version=1, **meta, boxes=boxes, switches=switches, windows=windows)
    path = os.path.join(out_dir, "bundle.json")
    text = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"), default=str)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(out_dir, "bundle.js"), "w", encoding="utf-8") as fh:
        fh.write("window.RAD_BUNDLE = " + text + ";\n")
    return path


def write_clip(path: str, paths: list[str], s0: int, s1: int, fps: float, max_width: int = 1280) -> str:
    """The raw frames s0..s1 as H.264 video (no drawing), for the browser page."""
    first = cv2.imread(paths[s0 - 1])
    h, w = first.shape[:2]
    s = min(1.0, max_width / w)
    size = (int(round(w * s)) // 2 * 2, int(round(h * s)) // 2 * 2)
    vw = VideoWriter(path, size, fps)
    try:
        for f in range(s0, s1 + 1):
            img = cv2.imread(paths[f - 1]) if f - 1 < len(paths) else None
            if img is None:
                img = np.zeros((size[1], size[0], 3), np.uint8)
            vw.write(cv2.resize(img, size, interpolation=cv2.INTER_AREA) if img.shape[1] != size[0] else img)
    finally:
        vw.close()
    return path


# ----------------------------------------------------------------------------- risk model
def add_risk_scores(sd, cfg, data_dir: str, train_glob: str | None, label: str = "y_risk") -> str:
    """Fill sd.cands['p_y_risk'] with a model trained on the other sequences. Returns a note."""
    from rad.dispatch.model import fit_full

    files = sorted(glob.glob(train_glob)) if train_glob else []
    files = [p for p in files if os.path.basename(os.path.dirname(p)) != sd.name]
    t = time.time()
    if files:
        train = pd.concat([pd.read_csv(p, low_memory=False) for p in files], ignore_index=True)
        src = f"{len(files)} sequence khác ({os.path.dirname(os.path.dirname(files[0]))})"
    else:  # build candidates of the other sequences (oracle detections, CPU)
        parts = []
        ign = dict(ignore_classes=cfg["gt_ignore_classes"], ignore_ioa=cfg["det_ignore_ioa"])
        for s in discover_sequences(data_dir):
            if s.name == sd.name or not s.gt_path:
                continue
            _, gv = load_gt(s, **ign)
            det = load_detections(s, "oracle", None, cfg["oracle_dets"], gt_vis=gv, **ign)
            if len(det):
                parts.append(prepare(s, det, cfg["bytetrack"], cfg["candidates"], **ign).cands)
                print(f"[risk] candidates of {s.name}: {len(parts[-1])}")
        train = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        src = f"{len(parts)} sequence khác (oracle detections, tính tại chỗ)"
    if len(train) == 0 or label not in train or train[label].sum() == 0:
        sd.cands["p_y_risk"] = -sd.cands["margin_det"].fillna(1.0)
        return "không có dữ liệu huấn luyện: dùng -margin làm điểm rủi ro"
    model = fit_full(train, label)
    sd.cands["p_y_risk"] = model.predict_proba(sd.cands[list(FEATURES)].astype(float).to_numpy())[:, 1]
    print(f"[risk] trained on {len(train)} candidates from {src} ({time.time() - t:.0f}s)")
    return f"mô hình rủi ro học trên {src}"


# ----------------------------------------------------------------------------- clip choice
def pick_event(sd, plans: dict, p: SMPParams, oracle_kw: dict) -> dict:
    """An ID switch of the baseline that a policy fixes (oracle VOS, whole sequence).

    Priority: fixed by RAD but not SMP > fixed by RAD > fixed by SMP > busiest window area.
    Among equals, the switch on the largest object (easiest to see)."""
    base = sd.cm.switches
    fixed = {}
    ran_frames = []
    for key, trig in plans.items():
        corrected, wins, _ = run_smp(sd.tracks, sd.log, make_backend("oracle", sd, **oracle_kw), trig, sd.length, p)
        after = clear_match(sd.gt, corrected)
        fx, _ = switch_fixed(base, after.switches)
        fixed[key] = fx.to_numpy().astype(bool) if len(base) else np.zeros(0, bool)
        ran_frames += [int(w.entry_frame) for w in wins if w.seed_frame is not None]
    if len(base):
        rad, smp = fixed.get("rad", np.zeros(len(base), bool)), fixed.get("smp", np.zeros(len(base), bool))
        gt_area = {}
        for r in sd.gt:
            gt_area[(int(r[F]), int(r[ID]))] = r[W] * r[H]
        for name, mask in (("RAD sửa, SMP không", rad & ~smp), ("RAD sửa", rad), ("SMP sửa", smp)):
            idx = np.flatnonzero(mask)
            if len(idx):
                areas = [gt_area.get((int(base.frame.iloc[i]), int(base.gt_id.iloc[i])), 0.0) for i in idx]
                i = int(idx[int(np.argmax(areas))])
                s = base.iloc[i]
                return dict(frame=int(s.frame), gt_id=int(s.gt_id), prev_trk=int(s.prev_trk), new_trk=int(s.new_trk),
                            reason=name)
    if ran_frames:
        fr = np.array(ran_frames)
        counts = [(np.abs(fr - c) <= 150).sum() for c in fr]
        return dict(frame=int(fr[int(np.argmax(counts))]), reason="nhiều cửa sổ nhất (không có lỗi nào được sửa)")
    return dict(frame=sd.length // 2, reason="giữa video")


def auto_crop(sd, event: dict, size: tuple, min_frac: float = 0.45, pad: float = 1.9, aspect: float = 16 / 9):
    Wimg, Himg = size
    ids = {event.get("prev_trk"), event.get("new_trk")} - {None}
    f = event["frame"]
    rows = []
    if ids:
        sel = np.isin(sd.tracks[:, ID].astype(int), list(ids)) & (np.abs(sd.tracks[:, F] - f) <= 45)
        rows.append(sd.tracks[sel])
    if "gt_id" in event:
        sel = (sd.gt[:, ID].astype(int) == event["gt_id"]) & (np.abs(sd.gt[:, F] - f) <= 45)
        rows.append(sd.gt[sel])
    rows = np.concatenate(rows) if rows else np.zeros((0, 9))
    if len(rows) == 0:
        return (0, 0, Wimg, Himg)
    x1, y1 = rows[:, X].min(), rows[:, Y].min()
    x2, y2 = (rows[:, X] + rows[:, W]).max(), (rows[:, Y] + rows[:, H]).max()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = max((x2 - x1) * pad, Wimg * min_frac), (y2 - y1) * pad
    if w / max(h, 1) > aspect:
        h = w / aspect
    else:
        w = h * aspect
    s = min(1.0, Wimg / w, Himg / h)
    w, h = w * s, h * s
    x0 = float(np.clip(cx - w / 2, 0, Wimg - w))
    y0 = float(np.clip(cy - h / 2, 0, Himg - h))
    return (x0, y0, w, h)


def parse_crop(text: str, sd, event: dict):
    if text == "none" or event is None and text == "auto":
        return (0, 0, sd.size[0], sd.size[1])
    if text == "auto":
        return auto_crop(sd, event, sd.size)
    x, y, w, h = [float(v) for v in text.split(",")]
    return (x, y, w, h)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq", default=None, help="default: first sequence with img1/ frames")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    ap.add_argument("--dets", choices=["oracle", "det", "dir"], default="oracle")
    ap.add_argument("--det-dir", default=None)
    ap.add_argument("--backend", choices=["oracle", "sam3"], default="oracle")
    ap.add_argument("--precision", choices=["auto", "fp16"], default=None, help="SAM 3 precision (else config)")
    ap.add_argument("--frames-on-gpu", action="store_true",
                    help="SAM 3: keep window frames in GPU memory instead of RAM (Colab RAM is 12.7 GB)")
    ap.add_argument("--train-cands", default=None,
                    help='glob of candidates.csv.gz from earlier runs, e.g. "$WORK/outputs/dispatch/*/candidates.csv.gz";'
                         " default: build them from the other sequences (oracle detections)")
    ap.add_argument("--tau", type=float, default=0.05, help="SMP margin threshold")
    ap.add_argument("--budget", type=float, default=1.0, help="RAD triggers = budget x number of SMP triggers")
    ap.add_argument("--start", type=int, default=None, help="first frame of the clip (default: automatic)")
    ap.add_argument("--frames", type=int, default=300, help="clip length when --start is given")
    ap.add_argument("--before", type=int, default=150, help="automatic clip: frames before the event")
    ap.add_argument("--after", type=int, default=150, help="automatic clip: frames after the event")
    ap.add_argument("--crop", default="auto", help="auto (around the event) | none | x,y,w,h")
    ap.add_argument("--layout", choices=["grid", "single"], default="grid")
    ap.add_argument("--single", choices=["base", "smp", "rad"], default="rad", help="panel for --layout single")
    ap.add_argument("--width", type=int, default=1920, help="output width (height = width x 9/16)")
    ap.add_argument("--fps", type=float, default=15.0, help="output fps (15 = half speed for 30 fps video)")
    ap.add_argument("--label-all", action="store_true", help="label every box (default: large or involved ones)")
    ap.add_argument("--no-video", action="store_true", help="only key frames and tables")
    ap.add_argument("--no-bundle", action="store_true", help="skip bundle.json / clip.mp4 for demo/index.html")
    ap.add_argument("--clip-width", type=int, default=1280, help="width of clip.mp4 (raw frames for the page)")
    ap.add_argument("--out", default="outputs/demo")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seqs = discover_sequences(args.data)
    if args.seq:
        seq = next((s for s in seqs if s.name == args.seq), None)
        if seq is None:
            sys.exit(f"no sequence named {args.seq} in {args.data}")
    else:
        seq = next((s for s in seqs if s.frame_paths()), None)
        if seq is None:
            sys.exit("no sequence has img1/ frames: run scripts/download_fasttracker.py --mode full --seqs <name>")
    paths = seq.frame_paths()
    if not paths:
        sys.exit(f"{seq.name}: no img1/ frames. Download them: "
                 f"python scripts/download_fasttracker.py --out <DS> --mode full --seqs {seq.name}")
    print("sequence:", seq.name)

    ign = dict(ignore_classes=cfg["gt_ignore_classes"], ignore_ioa=cfg["det_ignore_ioa"])
    _, gt_vis = load_gt(seq, **ign)
    det = load_detections(seq, args.dets, args.det_dir, cfg["oracle_dets"], gt_vis=gt_vis, **ign)
    if len(det) == 0:
        sys.exit(f"{seq.name}: 0 detections")
    t = time.time()
    sd = prepare(seq, det, cfg["bytetrack"], cfg["candidates"], **ign)
    has_gt = len(sd.gt) > 0
    print(f"[track] {len(det)} detections, {len(sd.tracks)} track boxes ({time.time() - t:.0f}s)"
          + ("" if has_gt else "; no ground truth: windows are shown, ID switches cannot be counted"))
    p = SMPParams(**{**asdict(cfg["smp"]), "image_size": sd.size})
    risk_note = add_risk_scores(sd, cfg, args.data, args.train_cands)

    smp_full = pol.smp_baseline(sd.log, args.tau, p.gap_G)
    n_rad = int(round(args.budget * max(len(smp_full), 1)))
    rad_full = pol.budget(sd.cands.dropna(subset=["p_y_risk"]), "p_y_risk", n_rad)
    plans = {"smp": smp_full, "rad": rad_full}
    print(f"[policy] whole sequence: SMP {len(smp_full)} triggers, RAD top-{n_rad} by risk")

    event = None
    if args.start is None and not has_gt:
        args.start = 1
    if args.start is None:
        event = pick_event(sd, plans, p, cfg["oracle_backend"])
        s0 = max(1, event["frame"] - args.before)
        s1 = min(sd.length, event["frame"] + args.after)
        print(f"[clip] event at frame {event['frame']} ({event['reason']}) -> frames {s0}-{s1}")
    else:
        s0 = max(1, args.start)
        s1 = min(sd.length, s0 + args.frames - 1)
        print(f"[clip] frames {s0}-{s1}")
    s1 = min(s1, len(paths))

    # --- run the windows of the clip with the chosen VOS backend
    kw = dict(cfg["oracle_backend"] if args.backend == "oracle" else cfg["sam3_backend"])
    if args.backend == "sam3" and args.precision:
        kw["precision"] = args.precision
    if args.backend == "sam3" and args.frames_on_gpu:
        kw["offload_video_to_cpu"] = False
    inner = make_backend(args.backend, sd, **kw)
    cache = CachingBackend(inner)
    margin = sd.log.groupby(["frame", "track_id"])["margin_det"].min() if len(sd.log) else pd.Series(dtype=float)
    risk = sd.cands.groupby(["frame", "track_id"])["p_y_risk"].max()
    runs = {}
    t = time.time()
    for key, trig in plans.items():
        clip = trig[(trig.frame >= s0) & (trig.frame <= s1)]
        rec = RecordingBackend(cache, verbose=args.backend == "sam3", label=key.upper())
        corrected, wins, stats = run_smp(sd.tracks, sd.log, rec, clip, sd.length, p)
        shapes = attach_shapes(wins, rec.calls)
        scores = {}
        for i, w in enumerate(wins):
            if w.witness_of is not None:
                continue
            k = (int(w.entry_frame), int(w.track_id))
            if key == "rad" and k in risk.index:
                scores[i] = f"rủi ro {risk[k]:.2f}"
            elif w.trigger == "gap":
                scores[i] = "tìm lại sau gap"
            elif k in margin.index:
                scores[i] = f"margin {margin[k]:.2f}"
        runs[key] = dict(tracks=corrected, windows=wins, shapes=shapes, scores=scores, stats=stats,
                         triggers=len(clip))
        print(f"[{key}] {len(clip)} triggers in clip -> {stats['windows']} windows, {stats['windows_ran']} ran, "
              f"{stats['vos_frames']} VOS frames, {stats['swaps_applied']} swaps applied")
    vos_s = time.time() - t
    gpu = inner.stats() if hasattr(inner, "stats") else {}
    print(f"[vos] {args.backend}: {getattr(inner, 'frames_processed', 0)} frames computed, "
          f"{cache.frames_replayed} replayed from cache ({vos_s:.0f}s)")

    # --- evaluation inside the clip
    def clip_rows(a):
        return a[(a[:, F] >= s0) & (a[:, F] <= s1)]

    gt_c = clip_rows(sd.gt)
    evals, switches, cms = {}, {}, {}
    shown = {"base": sd.tracks, "smp": runs["smp"]["tracks"], "rad": runs["rad"]["tracks"]}
    for key, trk in shown.items():
        if not has_gt:
            cms[key] = None
            switches[key] = pd.DataFrame(columns=["frame", "gt_id", "prev_trk", "new_trk"])
            evals[key] = None
            continue
        cm = clear_match(sd.gt, trk, frames=(s0, s1))
        cms[key] = cm
        switches[key] = cm.switches
        ev = evaluate(gt_c, clip_rows(trk))
        evals[key] = dict(idsw=int(cm.num_switches), idf1=round(float(ev["idf1"]), 4), mota=round(float(ev["mota"]), 4))
    print("[clip eval]", json.dumps(evals) if has_gt else "no ground truth")

    # --- panels
    det_name = DET_NAME[args.dets]
    vos_name = "SAM 3" if args.backend == "sam3" else "VOS"
    budget_txt = "cùng số lần kích hoạt với SMP" if abs(args.budget - 1) < 1e-9 else \
        f"ngân sách = {args.budget:g} x số kích hoạt của SMP"
    tau_txt = f"{args.tau:g}".replace(".", ",")
    panels = {
        "base": PanelData("base", "ByteTrack", f"chỉ tracker, {det_name}", sd.tracks,
                          switches=switches["base"], vos_name=vos_name, s0=s0, s1=s1),
        "smp": PanelData("smp", f"SMP (τ = {tau_txt})", f"gọi {vos_name} khi margin thấp hoặc track được tìm lại",
                         runs["smp"]["tracks"], runs["smp"]["windows"], runs["smp"]["shapes"], runs["smp"]["scores"],
                         switches["smp"], vos_name, s0, s1),
        "rad": PanelData("rad", "RAD (học khi nào gọi SAM)", f"gọi {vos_name} theo rủi ro dự đoán, {budget_txt}",
                         runs["rad"]["tracks"], runs["rad"]["windows"], runs["rad"]["shapes"], runs["rad"]["scores"],
                         switches["rad"], vos_name, s0, s1),
    }
    crop = parse_crop(args.crop, sd, event)
    focus = set()
    if event and "prev_trk" in event:
        focus = {event["prev_trk"], event["new_trk"]}
    for r in runs.values():
        for w in r["windows"]:
            if w.applied:
                focus |= {int(w.track_id), int(w.settled_id)}
    Wout = int(args.width) // 2 * 2
    Hout = int(round(Wout * 9 / 16)) // 2 * 2
    caption = (f"{seq.name} · khung {s0}–{s1} · {det_name} · VOS: "
               f"{'SAM 3' if args.backend == 'sam3' else 'oracle (bám GT, vẽ hộp thay cho mask)'}\n"
               f"RAD: {risk_note}; ID sửa được hiển thị từ khung swap (hiệu chỉnh offline)"
               + ("" if has_gt else "; video không có GT nên không đếm lỗi đổi ID"))

    out_dir = os.path.join(args.out, f"{seq.name}_{s0}-{s1}_{args.backend}")
    os.makedirs(os.path.join(out_dir, "keyframes"), exist_ok=True)
    if args.layout == "grid":
        pw, ph = Wout // 2, Hout // 2
        slots = [("base", 0, 0), ("smp", pw, 0), ("rad", 0, ph)]
    else:
        pw, ph = Wout, Hout
        slots = [(args.single, 0, 0)]
    view = View(crop, pw, ph)

    def render(f: int) -> np.ndarray:
        img = cv2.imread(paths[f - 1]) if f - 1 < len(paths) else None
        canvas = np.zeros((Hout, Wout, 3), np.uint8)
        texts: list[Text] = []
        for key, ox, oy in slots:
            draw_panel(canvas, texts, ox, oy, pw, ph, img, f, panels[key], view, label_all=args.label_all,
                       focus_ids=focus)
        if args.layout == "grid":
            draw_chart(canvas, texts, pw, ph, pw, ph, f, [panels["base"], panels["smp"], panels["rad"]], caption,
                       has_gt=has_gt)
            cv2.line(canvas, (pw, 0), (pw, Hout), (255, 255, 255), 2)
            cv2.line(canvas, (0, ph), (Wout, ph), (255, 255, 255), 2)
        return draw_texts(canvas, texts)

    # key frames: around the applied swaps of RAD (else SMP), else evenly spaced
    keys = []
    for key in ("rad", "smp"):
        for w in runs[key]["windows"]:
            if w.applied:
                keys += [int(w.entry_frame), int(w.swap_frame) + 2, min(int(w.swap_frame) + 30, s1)]
        if keys:
            break
    if event:
        keys += [max(event["frame"] - 20, s0), min(event["frame"] + 10, s1)]
    if not keys:
        keys = list(np.linspace(s0, s1, 4).astype(int))
    keys = sorted({k for k in keys if s0 <= k <= s1})
    for k in keys:
        cv2.imwrite(os.path.join(out_dir, "keyframes", f"frame_{k:05d}.png"), render(k))

    video = None
    if not args.no_video:
        video = os.path.join(out_dir, "demo.mp4")
        vw = VideoWriter(video, (Wout, Hout), args.fps)
        t = time.time()
        for f in range(s0, s1 + 1):
            vw.write(render(f))
        vw.close()
        print(f"[video] {video} ({s1 - s0 + 1} frames, {time.time() - t:.0f}s)")

    # tables
    for key in ("smp", "rad"):
        windows_frame(runs[key]["windows"]).to_csv(os.path.join(out_dir, f"windows_{key}.csv"), index=False)
    ev_rows = []
    for key, sw in switches.items():
        for s in sw.itertuples(index=False):
            ev_rows.append(dict(panel=key, frame=int(s.frame), event="idsw", gt_id=int(s.gt_id),
                                prev_trk=int(s.prev_trk), new_trk=int(s.new_trk)))
    for key in ("smp", "rad"):
        for w in runs[key]["windows"]:
            ev_rows.append(dict(panel=key, frame=int(w.entry_frame), event=f"window_{w.outcome.lower()}",
                                track=int(w.track_id), witness_of=w.witness_of, seed_frame=w.seed_frame,
                                end_frame=w.end_frame, vos_frames=int(w.frames), applied=bool(w.applied)))
    pd.DataFrame(ev_rows).sort_values(["frame", "panel"]).to_csv(os.path.join(out_dir, "events.csv"), index=False)
    bundle = clip_video = None
    if not args.no_bundle:
        t = time.time()
        clip_video = write_clip(os.path.join(out_dir, "clip.mp4"), paths, s0, s1, seq.fps, args.clip_width)
        meta = dict(sequence=seq.name, clip=[s0, s1], fps=float(seq.fps), size=[int(v) for v in sd.size],
                    has_gt=has_gt, dets=args.dets, dets_name=det_name, backend=args.backend, vos_name=vos_name,
                    tau=args.tau, budget=args.budget, risk_model=risk_note, event=event,
                    crop=[round(float(v), 1) for v in crop], clip_video="clip.mp4", created=time.strftime("%Y-%m-%d %H:%M"),
                    methods={k: dict(name=METHOD_NAME[k], title=panels[k].title, subtitle=panels[k].subtitle,
                                     eval=evals[k], triggers=(runs[k]["triggers"] if k in runs else 0),
                                     stats=({a: int(b) for a, b in runs[k]["stats"].items()} if k in runs else {}))
                             for k in ("base", "smp", "rad")})
        bundle = write_bundle(out_dir, meta, sd.gt if has_gt else None, shown, cms, runs, s0, s1)
        print(f"[bundle] {bundle} + clip.mp4 ({time.time() - t:.0f}s)")

    summary = dict(sequence=seq.name, clip=[s0, s1], dets=args.dets, backend=args.backend, event=event, has_gt=has_gt,
                   crop=[round(float(v), 1) for v in crop], risk_model=risk_note, tau=args.tau, budget=args.budget,
                   triggers_whole_sequence=dict(smp=len(smp_full), rad=n_rad),
                   clip_eval=evals, policies={k: {**{a: int(b) for a, b in r["stats"].items()}, "triggers": r["triggers"]}
                             for k, r in runs.items()},
                   vos_frames_computed=int(getattr(inner, "frames_processed", 0)),
                   vos_frames_replayed_from_cache=int(cache.frames_replayed), vos_seconds=round(vos_s, 1),
                   backend_stats={k: (round(v, 3) if isinstance(v, float) else v) for k, v in gpu.items()},
                   video=video, bundle=bundle, clip_video=clip_video,
                   keyframes=[f"keyframes/frame_{k:05d}.png" for k in keys])
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    print("[done]", out_dir)


if __name__ == "__main__":
    main()
