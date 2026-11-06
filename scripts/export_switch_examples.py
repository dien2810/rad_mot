#!/usr/bin/env python
"""Slide-ready clips of real identity switches, a few per type (fragment / swap / takeover).

Same setting as the 473 switches in the report: ByteTrack on oracle detections, switches
counted with CLEAR-MOT and typed by rad.pilot.analysis.classify_switches. Events are filtered
so that the clip is easy to read (one GT box per id, no jump of the GT box, the object fully
inside the image, a large object, daytime first; for fragment and takeover an object that was
occluded just before), cropped around the event and drawn simply:

  blue box   the track that followed the object before the switch ("ID a")
  orange box the track that follows it after the switch ("ID b")
  grey boxes other tracks in view
  white dashed box  the object while no track covers it
  red tag    "đổi ID a→b" from the switch frame on

Frames come from <seq>/img1, --zip-dir, or --remote (Hugging Face, HTTP range, needs a token
with access to the dataset). Output: <out>/<type>_<k>.mp4, a poster <type>_<k>.png and
examples.csv. Pick one clip per type by eye: a switch can still be an annotation error.

Example (Colab):
    python scripts/export_switch_examples.py --data $DATA --remote --out $OUT/switch_examples
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import spot_check_switches as sc  # noqa: E402
from rad.boxes import iou_matrix  # noqa: E402
from rad.config import load_config  # noqa: E402
from rad.io.mot import F, H, ID, V, W, X, Y, discover_sequences, group_by_frame  # noqa: E402
from rad.pilot.analysis import classify_switches  # noqa: E402
from rad.viz.demo import Text, VideoWriter, View, _dashed_rect, bgr, draw_texts  # noqa: E402

TYPE_VI = {"fragment": "Đứt mạch", "swap": "Hoán đổi", "takeover": "Chiếm ID"}
PREV = (30, 111, 217)
NEW = (217, 119, 6)
OTHER = (205, 211, 219)
RED = (194, 65, 45)
NAVY = (34, 53, 77)
PW, PH = 960, 540


def at_border(rows: np.ndarray, size: tuple, margin: float = 0.01) -> bool:
    """True if any of these GT boxes touches the image border (object entering or leaving)."""
    if len(rows) == 0:
        return False
    w_img, h_img = size
    m = margin * w_img
    return bool(((rows[:, X] < m) | (rows[:, Y] < m) | (rows[:, X] + rows[:, W] > w_img - m)
                 | (rows[:, Y] + rows[:, H] > h_img - m)).any())


def candidates(seq, tr: dict) -> pd.DataFrame:
    """Every switch of a sequence with the readability hints used to rank them.

    at_border        the object touches the image border within 3 frames of the switch: a new id
                     on an object that is leaving or entering, a poor example of the mechanism
    min_vis_before   lowest estimated visibility in the 30 frames before the switch (occlusion)
    """
    sw = classify_switches(tr["cm"], tr["tracks"])
    gt, vis = tr["gt"], tr["gt_vis"]
    size = (seq.width or 1920, seq.height or 1080)
    rows = []
    for ev in sw.to_dict("records"):
        f, gid = int(ev["frame"]), int(ev["gt_id"])
        mine = gt[gt[:, ID].astype(int) == gid]
        fr = mine[:, F].astype(int)
        me = mine[fr == f]
        vmine = vis[vis[:, ID].astype(int) == gid]
        vb = vmine[(vmine[:, F] >= f - 30) & (vmine[:, F] < f), V]
        rows.append({"seq": seq.name, "frame": f, "gt_id": gid, "prev_trk": int(ev["prev_trk"]),
                     "new_trk": int(ev["new_trk"]), "type": ev["type"],
                     "box_w": float(me[0, W]) if len(me) else 0.0, "box_h": float(me[0, H]) if len(me) else 0.0,
                     "at_border": at_border(mine[np.abs(fr - f) <= 3], size),
                     "min_vis_before": round(float(vb.min()), 2) if len(vb) else float("nan"),
                     **sc.gt_hints(gt, vis, gid, f)})
    return pd.DataFrame(rows)


def pick(pop: pd.DataFrame, per_type: int, min_h: float, types=("fragment", "swap", "takeover"),
         allow_border: bool = False) -> pd.DataFrame:
    """Readable events first: one GT box, no GT jump, inside the image, a large object, daytime,
    occluded just before (fragment, takeover), spread over sequences."""
    ok = (pop.gt_dup == 1) & (pop.box_h >= min_h) & (pop.gt_iou_min.isna() | (pop.gt_iou_min >= 0.5))
    ok &= ~((pop.type == "fragment") & (pop.gt_gap_before > 2))  # tracker error, not a hole in the labels
    if not allow_border:
        ok &= ~pop.at_border.astype(bool)
    pop = pop[ok].assign(day=pop.seq.str.contains("day").astype(int), area=pop.box_w * pop.box_h,
                         occluded=((pop.min_vis_before < 0.5) & pop.type.isin(["fragment", "takeover"])).astype(int))
    chosen = []
    for typ in types:
        sub = pop[pop.type == typ].sort_values(["occluded", "day", "area"], ascending=False)
        used_seq: dict = {}
        for r in sub.to_dict("records"):
            if sum(1 for c in chosen if c["type"] == typ) >= per_type:
                break
            if any(sc._near(r, c, 30) for c in chosen) or used_seq.get(r["seq"], 0) >= 2:
                continue
            used_seq[r["seq"]] = used_seq.get(r["seq"], 0) + 1
            chosen.append(r)
    return pd.DataFrame(chosen, columns=list(pop.columns))


def _tag(texts, x, y, s, bg, size=22, anchor="ld"):
    texts.append(Text(x, y, s, size, (251, 252, 254), bg, True, anchor=anchor))


def frame_image(img, g: int, ev: dict, ctx: dict, crop: tuple, tag_until: int, kind: str) -> np.ndarray:
    view = View(crop, PW, PH)
    panel = view.image(img)
    texts: list[Text] = []
    prev, new, gid, f = int(ev["prev_trk"]), int(ev["new_trk"]), int(ev["gt_id"]), int(ev["frame"])
    x0, y0, cw, ch = crop
    rows = ctx["trk_by"].get(g, np.zeros((0, 9)))
    boxes = {}
    for r in rows:
        bb = sc._xyxy(r)[0]
        if bb[2] < x0 or bb[0] > x0 + cw or bb[3] < y0 or bb[1] > y0 + ch:
            continue
        tid = int(r[ID])
        p = view.box(bb)
        if tid in (prev, new):
            boxes[tid] = (bb, p)
        else:
            cv2.rectangle(panel, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), bgr(OTHER), 2)
    for tid, col in ((prev, PREV), (new, NEW)):
        if tid in boxes:
            p = boxes[tid][1]
            cv2.rectangle(panel, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), bgr(col), 4)
            _tag(texts, p[0], p[1] - 4, f"ID {tid}", col)
    me = ctx["gt_by"].get(g, np.zeros((0, 9)))
    me = me[me[:, ID].astype(int) == gid]
    if len(me):
        gb = sc._xyxy(me)[0]
        covered = any(iou_matrix(gb[None], b[0][None])[0, 0] >= 0.3 for b in boxes.values())
        if not covered:
            _dashed_rect(panel, view.box(gb), (251, 252, 254), 3, 10)
    if f <= g <= tag_until and new in boxes:
        p = boxes[new][1]
        below = p[3] + 34 < PH
        _tag(texts, p[0], p[3] + 6 if below else p[1] - 34, f"đổi ID {prev}→{new}", RED, anchor="la" if below else "ld")
    _tag(texts, 14, 14, TYPE_VI.get(kind, kind), NAVY, 24, anchor="la")
    _tag(texts, PW - 14, 14, f"khung {g}", (20, 33, 48), 22, anchor="ra")
    return draw_texts(panel, texts)


def export(ev: dict, tr: dict, src, size: tuple, fps: float, path: str, poster: str,
           pre_s: float, post_s: float, speed: float) -> tuple:
    f = int(ev["frame"])
    f0 = max(1, f - int(round(pre_s * fps)))
    f1 = min(tr["length"], f + int(round(post_s * fps)))
    crop = sc.event_crop(tr["gt"], tr["tracks"], ev, f0, f1, size, pad=2.2, min_frac=0.3)
    ctx = dict(gt_by=group_by_frame(tr["gt"]), trk_by=group_by_frame(tr["tracks"]))
    tag_until = f + int(round(1.0 * fps))
    shot = min(f1, f + int(round(0.4 * fps)))
    vw = VideoWriter(path, (PW, PH), max(1.0, fps * speed), crf=20)
    try:
        for g in range(f0, f1 + 1):
            im = frame_image(src.get(g) if src is not None else None, g, ev, ctx, crop, tag_until, ev["type"])
            vw.write(im)
            if g == shot:
                cv2.imwrite(poster, im)
    finally:
        vw.close()
    return f0, f1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="split folder with one sub-folder per sequence (gt/gt.txt, img1/)")
    ap.add_argument("--config", default=os.path.join(HERE, "..", "configs", "default.yaml"))
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--out", default="outputs/switch_examples")
    ap.add_argument("--per-type", type=int, default=3, help="clips per switch type")
    ap.add_argument("--min-h", type=float, default=60, help="minimum box height (px) of the object")
    ap.add_argument("--pre", type=float, default=1.5, help="seconds before the switch")
    ap.add_argument("--post", type=float, default=1.5, help="seconds after the switch")
    ap.add_argument("--speed", type=float, default=0.5, help="playback speed (0.5 = half speed)")
    ap.add_argument("--types", nargs="*", default=["fragment", "swap", "takeover"],
                    choices=["fragment", "swap", "takeover"], help="switch types to export")
    ap.add_argument("--allow-border", action="store_true", help="also pick objects touching the image border")
    ap.add_argument("--remote", action="store_true", help="read frames from the Hugging Face zips")
    ap.add_argument("--zip-dir", default=None, help="folder with downloaded <sequence>.zip files")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seqs = discover_sequences(args.data)
    if args.seqs:
        seqs = [s for s in seqs if s.name in set(args.seqs)]
    os.makedirs(args.out, exist_ok=True)
    tracked, pops = {}, []
    for s in seqs:
        tr = sc.track_oracle(s, cfg)
        cand = candidates(s, tr)
        tracked[s.name] = (s, tr)
        if len(cand) and sc.has_frames(s, args.remote, args.zip_dir):
            pops.append(cand)
        print(f"[track] {s.name}: {len(cand)} switches", flush=True)
    if not pops:
        sys.exit("no switch in a sequence with frames; add --remote or --zip-dir")
    pop = pd.concat(pops, ignore_index=True)
    pop.to_csv(os.path.join(args.out, "switches_all.csv"), index=False)
    print("[all] by type:", pop.type.value_counts().to_dict())
    edge = pop.groupby("type").at_border.agg(["sum", "count"])
    print("[all] object at the image border (±3 frames):",
          {t: f"{int(r['sum'])}/{int(r['count'])}" for t, r in edge.iterrows()},
          f"total {int(pop.at_border.sum())}/{len(pop)}")
    sel = pick(pop, args.per_type, args.min_h, tuple(args.types), args.allow_border)
    if sel.empty:
        sys.exit("no readable switch; lower --min-h")
    rows = []
    for seq_name, evs in sel.groupby("seq", sort=True):
        s, tr = tracked[seq_name]
        src, how = sc.frame_source(s, args.remote, args.zip_dir)
        size = (s.width or 1920, s.height or 1080)
        try:
            for ev in evs.to_dict("records"):
                k = sum(1 for r in rows if r["type"] == ev["type"]) + 1
                name = f"{ev['type']}_{k}"
                f0, f1 = export(ev, tr, src, size, s.fps, os.path.join(args.out, name + ".mp4"),
                                os.path.join(args.out, name + ".png"), args.pre, args.post, args.speed)
                rows.append({"clip": name + ".mp4", **ev, "clip_from": f0, "clip_to": f1, "frames_from": how})
                print(f"  {name}: {seq_name} frame {ev['frame']} ID {ev['prev_trk']}→{ev['new_trk']}", flush=True)
        finally:
            if src is not None:
                src.close()
    out = pd.DataFrame(rows).sort_values("clip")
    out.to_csv(os.path.join(args.out, "examples.csv"), index=False)
    print(f"\n{len(out)} clips in {args.out}; see examples.csv. Pick one clip per type by eye.")


if __name__ == "__main__":
    main()
