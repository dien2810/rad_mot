#!/usr/bin/env python
"""Spot-check identity switches: tracker error or annotation error?

In the oracle setting (detections = ground truth), every CLEAR-MOT identity switch is
blamed on the tracker. If the ground-truth IDs themselves are wrong (an ID jumps to
another vehicle, one vehicle gets a new ID), those "switches" are label noise. This
script draws a random sample of switches and makes them quick to judge by eye.

Step 1, sample and render (CPU; frames are needed only for the sampled events):
    python scripts/spot_check_switches.py --data datasets/FastTracker/train --remote \
        --out outputs/spot_check --n 30
  - runs ByteTrack on oracle detections exactly as run_pilot.py --dets oracle does and
    lists every switch with its type (fragment / swap / takeover),
  - draws a random sample, stratified by type in proportion to the type counts, so the
    overall share needs no re-weighting (near-duplicates of the same incident are skipped),
  - writes per event clips/<id>.mp4 (about +-1 s around the switch, cropped, slowed down)
    and sheets/<id>.jpg (4 frames, 2 x 2), plus labels.csv, switches_all.csv and index.html.
  Frames come from <seq>/img1 when it exists. Otherwise --remote reads only the needed
  JPEGs out of the Hugging Face zip with HTTP Range requests (about 20-40 MB per event),
  or --zip-dir reads them from downloaded <seq>.zip files. Without a frame source a
  sequence is left out of the sample (and the report says so).

Step 2, label: open index.html in a browser (or edit labels.csv) and choose per event
    tracker  the tracker changed identity; the annotation is consistent
    gt_id    annotation identity error: a GT id jumps to another object, or one object
             gets a new GT id
    gt_box   other annotation error (missing, misplaced or duplicate box) that creates
             the switch
    unclear  cannot tell
  then press the export button (saves labels.csv) and put the file back in the folder.

Step 3, summarise:
    python scripts/spot_check_switches.py --summarize outputs/spot_check/labels.csv
  -> share of annotation-caused switches (gt_id + gt_box) with a Wilson 95% interval,
     per switch type, and the implied number among all switches of the sampled videos.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
import zipfile
from dataclasses import asdict
from typing import Optional

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rad.boxes import iou_matrix  # noqa: E402
from rad.config import load_config  # noqa: E402
from rad.eval.idsw import clear_match  # noqa: E402
from rad.io.mot import F, H, ID, IMAGE_EXTS, V, W, X, Y, discover_sequences, group_by_frame  # noqa: E402
from rad.pilot.analysis import classify_switches  # noqa: E402
from rad.pipeline import load_detections, load_gt  # noqa: E402
from rad.tracker.bytetrack import ByteTrackParams, run_bytetrack  # noqa: E402
from rad.viz.demo import GRID, IDSW, INK, MUTED, Text, VideoWriter, View, _dashed_rect, bgr, draw_texts  # noqa: E402

REPO = "Hamidreza-Hashemp/FastTracker-Benchmark"
LABELS = ("tracker", "gt_id", "gt_box", "unclear")
ANNOTATION = ("gt_id", "gt_box")
TYPE_VI = {"fragment": "đứt track", "swap": "hoán đổi", "takeover": "chiếm ID"}
PREV = (47, 111, 189)    # track before the switch (RGB)
NEW = (212, 118, 44)     # track after the switch
FOCUS = (255, 255, 255)  # the GT object that switched
HEADER = 64              # header height in px at 960 px width


# ---------------------------------------------------------------------------- tracking
def track_oracle(seq, cfg: dict) -> dict:
    """ByteTrack on oracle detections, as in run_pilot.py --dets oracle (no candidates)."""
    ign = dict(ignore_classes=cfg["gt_ignore_classes"], ignore_ioa=cfg["det_ignore_ioa"])
    gt, gt_vis = load_gt(seq, **ign)
    det = load_detections(seq, "oracle", None, cfg["oracle_dets"], gt_vis=gt_vis, **ign)
    length = int(seq.length or (gt[:, F].max() if len(gt) else 0))
    bt = ByteTrackParams(**{**asdict(cfg["bytetrack"]), "frame_rate": seq.fps})
    tracks, _ = run_bytetrack(det, length, bt)
    cm = clear_match(gt, tracks)
    return dict(gt=gt, gt_vis=gt_vis, tracks=tracks, cm=cm, length=length)


def _xyxy(r: np.ndarray) -> np.ndarray:
    r = np.atleast_2d(r)
    return np.stack([r[:, X], r[:, Y], r[:, X] + r[:, W], r[:, Y] + r[:, H]], 1)


def gt_hints(gt: np.ndarray, gt_vis: np.ndarray, gt_id: int, f: int, window: int = 5) -> dict:
    """Cheap hints for the reviewer (not a verdict).

    gt_gap_before  longest stretch of frames without this GT id that ended in the 10 frames
                   before the switch (0 = annotated continuously; large = object or label missing)
    gt_iou_min     lowest IoU between consecutive annotated boxes of this GT id in [f-5, f+5];
                   a value near 0 means the box jumped, often an ID moved to another object
    gt_vis         estimated visibility of the GT box at the switch
    n_gt_overlap   other GT boxes overlapping it (IoU > 0.1) at the switch
    gt_dup         number of boxes with this GT id in the switch frame (> 1 is an annotation error)
    """
    rows = gt[gt[:, ID].astype(int) == gt_id]
    rows = rows[np.argsort(rows[:, F])]
    fr = rows[:, F].astype(int)
    upto = fr[fr <= f]
    gaps = [int(b - a - 1) for a, b in zip(upto[:-1], upto[1:]) if b >= f - 10 and b - a > 1]
    gap = max(gaps) if gaps else 0
    sel = rows[(fr >= f - window) & (fr <= f + window)]
    ious = []
    for a, b in zip(sel[:-1], sel[1:]):
        if int(b[F]) - int(a[F]) == 1:
            ious.append(float(iou_matrix(_xyxy(a), _xyxy(b))[0, 0]))
    vis = gt_vis[(gt_vis[:, ID].astype(int) == gt_id) & (gt_vis[:, F].astype(int) == f)]
    at = gt[gt[:, F].astype(int) == f]
    me = at[at[:, ID].astype(int) == gt_id]
    others = at[at[:, ID].astype(int) != gt_id]
    n_ov = int((iou_matrix(_xyxy(me), _xyxy(others))[0] > 0.1).sum()) if len(me) and len(others) else 0
    return dict(gt_gap_before=gap, gt_iou_min=round(min(ious), 3) if ious else float("nan"),
                gt_vis=round(float(vis[0, V]), 2) if len(vis) else float("nan"), n_gt_overlap=n_ov,
                gt_dup=int(len(me)))


# ---------------------------------------------------------------------------- sampling
def allocate(counts: dict, n: int) -> dict:
    """Proportional allocation with largest remainders (sum = min(n, total))."""
    total = sum(counts.values())
    n = min(n, total)
    if total == 0 or n == 0:
        return {k: 0 for k in counts}
    raw = {k: n * v / total for k, v in counts.items()}
    out = {k: int(math.floor(r)) for k, r in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: n - sum(out.values())]:
        out[k] += 1
    return out


def _near(a, b, frames: int = 10) -> bool:
    if a["seq"] != b["seq"] or abs(int(a["frame"]) - int(b["frame"])) > frames:
        return False
    ta = {int(a["prev_trk"]), int(a["new_trk"])}
    tb = {int(b["prev_trk"]), int(b["new_trk"])}
    return bool(ta & tb) or int(a["gt_id"]) == int(b["gt_id"])


def sample_switches(pop: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Stratified (by type) random sample; skips events of an incident already chosen."""
    rng = np.random.default_rng(seed)
    alloc = allocate(pop.groupby("type").size().to_dict(), n)
    chosen: list[dict] = []
    for typ in sorted(alloc):
        sub = pop[pop.type == typ]
        k = 0
        for i in rng.permutation(len(sub)):
            if k >= alloc[typ]:
                break
            r = sub.iloc[i].to_dict()
            if any(_near(r, c) for c in chosen):
                continue
            chosen.append(r)
            k += 1
    out = pd.DataFrame(chosen, columns=pop.columns)
    return out.sort_values(["seq", "frame"]).reset_index(drop=True)


# ---------------------------------------------------------------------------- frames
def _frame_key(name: str):
    stem = os.path.splitext(os.path.basename(name))[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    return (int(digits) if digits else 0, name)


class LocalFrames:
    def __init__(self, paths: list[str]):
        self.paths = paths

    def get(self, f: int) -> Optional[np.ndarray]:
        return cv2.imread(self.paths[f - 1]) if 1 <= f <= len(self.paths) else None

    def close(self) -> None:
        pass


class ZipFrames:
    """Frames read from a sequence zip (local path or any seekable file object)."""

    def __init__(self, src):
        self.zf = zipfile.ZipFile(src)
        names = [n for n in self.zf.namelist()
                 if n.lower().endswith(IMAGE_EXTS) and ("/img1/" in "/" + n.replace("\\", "/"))
                 and "__MACOSX" not in n]
        self.names = sorted(names, key=_frame_key)

    def get(self, f: int) -> Optional[np.ndarray]:
        if not 1 <= f <= len(self.names):
            return None
        buf = np.frombuffer(self.zf.read(self.names[f - 1]), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def close(self) -> None:
        self.zf.close()


_REPO_FILES: Optional[list] = None


def open_remote(seq_name: str):
    """ZipFrames over the Hugging Face zip of a sequence, via HTTP Range requests."""
    global _REPO_FILES
    from huggingface_hub import HfApi, hf_hub_url

    from rad.io.remote_zip import HttpRangeFile
    try:
        from huggingface_hub import get_token
    except ImportError:  # older huggingface_hub
        def get_token():
            return os.environ.get("HF_TOKEN")
    if _REPO_FILES is None:
        _REPO_FILES = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.endswith(".zip")]
    match = [f for f in _REPO_FILES if os.path.splitext(os.path.basename(f))[0] == seq_name]
    if not match:
        raise FileNotFoundError(f"{seq_name}.zip not found in {REPO}")
    token = get_token()
    rf = HttpRangeFile(hf_hub_url(REPO, match[0], repo_type="dataset"),
                       headers={"Authorization": f"Bearer {token}"} if token else {})
    return ZipFrames(rf), rf


def frame_source(seq, remote: bool, zip_dir: Optional[str]):
    """(source, description) or (None, reason)."""
    paths = seq.frame_paths()
    if paths:
        return LocalFrames(paths), "img1"
    if zip_dir:
        p = os.path.join(zip_dir, f"{seq.name}.zip")
        if os.path.exists(p):
            return ZipFrames(p), "zip"
    if remote:
        src, _ = open_remote(seq.name)
        return src, "remote"
    return None, "no frames"


def has_frames(seq, remote: bool, zip_dir: Optional[str]) -> bool:
    return bool(seq.frame_paths()) or remote or bool(zip_dir and os.path.exists(os.path.join(zip_dir, f"{seq.name}.zip")))


# ---------------------------------------------------------------------------- drawing
def event_crop(gt, tracks, ev: dict, f0: int, f1: int, size: tuple, pad: float = 1.8,
               min_frac: float = 0.25, aspect: float = 16 / 9) -> tuple:
    Wimg, Himg = size
    rows = [gt[(gt[:, ID].astype(int) == int(ev["gt_id"])) & (gt[:, F] >= f0) & (gt[:, F] <= f1)]]
    ids = [int(ev["prev_trk"]), int(ev["new_trk"])]
    near = (np.abs(tracks[:, F] - int(ev["frame"])) <= max(5, (f1 - f0) // 4))
    rows.append(tracks[np.isin(tracks[:, ID].astype(int), ids) & near])
    rows = np.concatenate(rows)
    if len(rows) == 0:
        return (0.0, 0.0, float(Wimg), float(Himg))
    b = _xyxy(rows)
    x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = max((x2 - x1) * pad, Wimg * min_frac), max((y2 - y1) * pad, 1.0)
    if w / h > aspect:
        h = w / aspect
    else:
        w = h * aspect
    s = min(1.0, Wimg / w, Himg / h)
    w, h = w * s, h * s
    return (float(np.clip(cx - w / 2, 0, Wimg - w)), float(np.clip(cy - h / 2, 0, Himg - h)), w, h)


def render(img, f: int, ev: dict, ctx: dict, crop: tuple, pw: int = 960, ph: int = 540) -> np.ndarray:
    """One annotated frame: header (event info) above the cropped, enlarged image.

    Full size (960 px) for the clip; small panels (480 px) for the 4-frame sheet, where
    the header keeps only what changes from frame to frame.
    """
    k = pw / 960
    small = k < 0.75
    hh = int(round(HEADER * k)) // 2 * 2

    def fs(v: float) -> int:
        return max(int(round(v * k)), 11)

    view = View(crop, pw, ph)
    panel = view.image(img)
    texts: list[Text] = []
    gid, prev, new = int(ev["gt_id"]), int(ev["prev_trk"]), int(ev["new_trk"])
    thick = max(2, int(round(3 * k)))
    x0, y0, cw, ch = crop
    focus = []
    for r in ctx["gt_by"].get(f, np.zeros((0, 9))):
        bb = _xyxy(r)[0]
        if bb[2] < x0 or bb[0] > x0 + cw or bb[3] < y0 or bb[1] > y0 + ch:
            continue
        p = view.box(bb)
        if int(r[ID]) == gid:
            focus.append(p)
        else:
            cv2.rectangle(panel, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), bgr(GRID), 1)
            if not small:
                texts.append(Text(p[0] + 2, p[1] + 2, f"g{int(r[ID])}", fs(11), GRID))
    for tid, col, name in ((prev, PREV, "trước"), (new, NEW, "sau")):
        rows = ctx["trk_by"].get(f, np.zeros((0, 9)))
        for r in rows[rows[:, ID].astype(int) == tid]:
            p = view.box(_xyxy(r)[0])
            cv2.rectangle(panel, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), bgr(col), thick)
            texts.append(Text(p[0], p[1] - 3 * k, f"T{tid}" if small else f"T{tid} ({name})", fs(15),
                              (255, 255, 255), col, True, anchor="ld"))
    for p in focus:  # drawn last so the dashed GT box stays visible over the track box
        _dashed_rect(panel, p, bgr(FOCUS), thick, max(int(10 * k), 5))
        texts.append(Text(p[0], p[3] + 3 * k, f"GT {gid}", fs(15), INK, FOCUS, True))
    if f == int(ev["frame"]):
        cv2.rectangle(panel, (0, 0), (pw - 1, ph - 1), bgr(IDSW), max(4, int(6 * k)))
    canvas = np.full((ph + hh, pw, 3), 250, np.uint8)
    canvas[hh:] = panel
    texts = [Text(t.x, t.y + hh, t.s, t.size, t.fill, t.bg, t.bold, t.anchor) for t in texts]
    m = ctx["match"].get(f, {}).get(gid)
    now = f"GT {gid} khớp T{m}" if m is not None else f"GT {gid}: không khớp track nào"
    dt = f - int(ev["frame"])
    if small:
        texts.append(Text(8, 5, f"khung {f} ({dt:+d})", 13, INK, bold=True))
        texts.append(Text(pw - 8, 5, now, 12, MUTED, anchor="ra"))
    else:
        texts.append(Text(10, 8, f"#{ev['event_id']}  {ev['seq']}  ·  khung {f} ({dt:+d})  ·  "
                          f"{TYPE_VI.get(ev['type'], ev['type'])}: GT {gid} T{prev} → T{new}", 16, INK, bold=True))
        texts.append(Text(10, 36, f"{now}   |   trắng nét đứt: GT {gid}   xanh: T{prev}   cam: T{new}   "
                          f"xám: GT khác   viền đỏ: khung đổi ID", 13, MUTED))
    return draw_texts(canvas, texts)


def make_context(tr: dict, gid: int) -> dict:
    m = tr["cm"].matches
    m = m[m.gt_id == gid]
    return dict(gt_by=group_by_frame(tr["gt"]), trk_by=group_by_frame(tr["tracks"]),
                match={int(f): {gid: int(t)} for f, t in zip(m.frame, m.trk_id)})


def render_event(ev: dict, tr: dict, src, size: tuple, fps: float, out: str, pre_s: float, post_s: float,
                 speed: float, ctx: Optional[dict] = None) -> dict:
    f = int(ev["frame"])
    pre, post = max(1, int(round(pre_s * fps))), max(1, int(round(post_s * fps)))
    f0, f1 = max(1, f - pre), min(tr["length"], f + post)
    crop = event_crop(tr["gt"], tr["tracks"], ev, f0, f1, size)
    ctx = ctx or make_context(tr, int(ev["gt_id"]))
    clip = os.path.join("clips", f"{ev['event_id']}.mp4")
    sheet = os.path.join("sheets", f"{ev['event_id']}.jpg")
    frames: dict[int, np.ndarray] = {}
    vw = VideoWriter(os.path.join(out, clip), (960, 540 + HEADER), max(1.0, fps * speed))
    try:
        for g in range(f0, f1 + 1):
            img = src.get(g) if src is not None else None
            fr = render(img, g, ev, ctx, crop)
            vw.write(fr)
            if g in (f0, f - 1, f, f1):
                frames[g] = render(img, g, ev, ctx, crop, 480, 270)
    finally:
        vw.close()
    keys = [g for g in (f0, f - 1, f, f1) if g in frames]
    keys = list(dict.fromkeys(keys))
    tiles = [frames[g] for g in keys]
    if len(tiles) == 4:  # 2 x 2 grid reads better in a browser than a 1 x 4 strip
        strip = np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)
    else:
        strip = np.concatenate(tiles, 1)
    os.makedirs(os.path.join(out, "sheets"), exist_ok=True)
    cv2.imwrite(os.path.join(out, sheet), strip, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return dict(clip=clip.replace(os.sep, "/"), sheet=sheet.replace(os.sep, "/"), clip_from=f0, clip_to=f1)


# ---------------------------------------------------------------------------- html
def write_index(out: str, events: pd.DataFrame) -> str:
    rows = events.to_dict("records")
    data = json.dumps([{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
                       for r in rows], ensure_ascii=False, default=str)
    cards = []
    for r in rows:
        hint = (f"GT vắng {r['gt_gap_before']} khung ngay trước lỗi · IoU GT liên tiếp nhỏ nhất {r['gt_iou_min']} · "
                f"độ hiển thị {r['gt_vis']} · {r['n_gt_overlap']} hộp GT chồng lấn"
                + (f" · {r['gt_dup']} hộp cùng ID GT trong khung" if int(r.get('gt_dup', 1) or 1) > 1 else ""))
        opts = "".join(f'<label><input type="radio" name="l{r["event_id"]}" value="{v}"> {v}</label>' for v in LABELS)
        cards.append(f"""
<section class="card" id="e{r['event_id']}">
  <h2>#{r['event_id']} · {html.escape(str(r['seq']))} · khung {r['frame']} · {TYPE_VI.get(r['type'], r['type'])}
      · GT {r['gt_id']}: T{r['prev_trk']} → T{r['new_trk']}</h2>
  <p class="hint">{html.escape(hint)}</p>
  <img src="{r['sheet']}" alt="4 khung quanh lỗi #{r['event_id']}" loading="lazy">
  <details><summary>Xem clip (tốc độ {r.get('speed', '')}×)</summary>
    <video src="{r['clip']}" controls preload="none" muted></video></details>
  <div class="row">{opts}<input class="note" placeholder="ghi chú" data-id="{r['event_id']}"></div>
</section>""")
    page = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kiểm tra nhãn lỗi đổi ID</title>
<style>
:root {{ --ink:#1f2328; --muted:#656d76; --line:#d0d7de; --bg:#f6f8fa; --card:#fff; --accent:#2f6fbd; }}
@media (prefers-color-scheme: dark) {{ :root {{ --ink:#e6edf3; --muted:#9198a1; --line:#3d444d; --bg:#0d1117;
  --card:#161b22; --accent:#4c8fe0; }} }}
body {{ margin:0; padding:16px; background:var(--bg); color:var(--ink); font:15px/1.45 system-ui,sans-serif; }}
header {{ position:sticky; top:0; background:var(--bg); padding:8px 0 12px; border-bottom:1px solid var(--line);
  z-index:1; }}
h1 {{ font-size:20px; margin:0 0 6px; }} h2 {{ font-size:16px; margin:0 0 4px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; margin:14px 0; }}
.card img, .card video {{ width:100%; height:auto; border-radius:4px; display:block; margin:6px 0; }}
.hint, .legend {{ color:var(--muted); font-size:13px; margin:0; }}
.row {{ display:flex; flex-wrap:wrap; gap:14px; align-items:center; margin-top:6px; }}
.note {{ flex:1; min-width:180px; padding:4px 6px; background:var(--card); color:var(--ink);
  border:1px solid var(--line); border-radius:4px; }}
button {{ background:var(--accent); color:#fff; border:0; border-radius:6px; padding:7px 14px; font-weight:600;
  cursor:pointer; }}
.done {{ border-color:var(--accent); }}
</style></head><body>
<header><h1>Kiểm tra nhãn của {len(rows)} lỗi đổi ID (thiết lập oracle)</h1>
<p class="legend"><b>tracker</b>: tracker đổi ID thật, nhãn nhất quán · <b>gt_id</b>: nhãn ID sai (ID GT nhảy sang
xe khác, hoặc một xe nhận ID GT mới) · <b>gt_box</b>: lỗi hộp nhãn khác (thiếu, lệch, trùng) gây ra lỗi ·
<b>unclear</b>: không kết luận được</p>
<p class="row"><span id="progress"></span><button id="export">Tải labels.csv</button></p></header>
{''.join(cards)}
<script>
const EVENTS = {data};
const KEY = 'spotcheck:' + location.pathname;
let state = {{}};
try {{ state = JSON.parse(localStorage.getItem(KEY) || '{{}}'); }} catch (e) {{ state = {{}}; }}
function save() {{ try {{ localStorage.setItem(KEY, JSON.stringify(state)); }} catch (e) {{}} }}
function refresh() {{
  let n = 0;
  for (const ev of EVENTS) {{
    const s = state[ev.event_id] || {{}};
    const card = document.getElementById('e' + ev.event_id);
    if (s.label) {{ n++; card.classList.add('done'); }} else card.classList.remove('done');
  }}
  document.getElementById('progress').textContent = `Đã gán ${{n}}/${{EVENTS.length}}`;
}}
for (const ev of EVENTS) {{
  const s = state[ev.event_id] || {{}};
  document.querySelectorAll(`input[name="l${{ev.event_id}}"]`).forEach(r => {{
    if (r.value === s.label) r.checked = true;
    r.addEventListener('change', () => {{ state[ev.event_id] = {{...(state[ev.event_id] || {{}}), label: r.value}};
      save(); refresh(); }});
  }});
  const note = document.querySelector(`.note[data-id="${{ev.event_id}}"]`);
  note.value = s.note || '';
  note.addEventListener('input', () => {{ state[ev.event_id] = {{...(state[ev.event_id] || {{}}), note: note.value}};
    save(); }});
}}
function csvCell(v) {{ v = (v === null || v === undefined) ? '' : String(v);
  return /[",\\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }}
document.getElementById('export').addEventListener('click', () => {{
  const cols = Object.keys(EVENTS[0]).filter(c => c !== 'label' && c !== 'note').concat(['label', 'note']);
  const lines = [cols.join(',')];
  for (const ev of EVENTS) {{
    const s = state[ev.event_id] || {{}};
    const row = {{...ev, label: s.label || '', note: s.note || ''}};
    lines.push(cols.map(c => csvCell(row[c])).join(','));
  }}
  const blob = new Blob([lines.join('\\n') + '\\n'], {{type: 'text/csv'}});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'labels.csv';
  document.body.appendChild(a); a.click(); a.remove();
}});
refresh();
</script></body></html>
"""
    path = os.path.join(out, "index.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)
    return path


# ---------------------------------------------------------------------------- summary
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def summarize(labels_csv: str, population_csv: Optional[str] = None) -> str:
    lab = pd.read_csv(labels_csv)
    lab["label"] = lab["label"].fillna("").astype(str).str.strip().str.lower()
    bad = lab[~lab.label.isin(LABELS)]
    ok = lab[lab.label.isin(LABELS)]
    decided = ok[ok.label != "unclear"]
    k = int(decided.label.isin(ANNOTATION).sum())
    n = len(decided)
    p, lo, hi = wilson(k, n)
    lines = ["# Kiểm tra nhãn lỗi đổi ID", "",
             f"- Sự kiện trong mẫu: {len(lab)}; đã gán nhãn hợp lệ: {len(ok)}; chưa gán/không hợp lệ: {len(bad)}",
             "- Nhãn: " + ", ".join(f"{v} {int((ok.label == v).sum())}" for v in LABELS)]
    if n:
        lines.append(f"- Tỷ lệ lỗi do nhãn (gt_id + gt_box, bỏ unclear): {k}/{n} = {p:.0%} "
                     f"(khoảng Wilson 95%: {lo:.0%}–{hi:.0%})")
        n_unc = int((ok.label == "unclear").sum())
        if n_unc:
            p2 = (k + n_unc) / (n + n_unc)
            lines.append(f"- Nếu mọi ca unclear đều do nhãn (cận trên): {k + n_unc}/{n + n_unc} = {p2:.0%}")
    pop_path = population_csv or os.path.join(os.path.dirname(os.path.abspath(labels_csv)), "switches_all.csv")
    if n and os.path.exists(pop_path):
        pop = pd.read_csv(pop_path)
        if "in_population" in pop:
            pop = pop[pop.in_population.astype(bool)]
        N = len(pop)
        lines.append(f"- Tổng số lỗi đổi ID trong các video được lấy mẫu: {N}; ước tính số lỗi do nhãn: "
                     f"{p * N:.0f} (khoảng {lo * N:.0f}–{hi * N:.0f})")
        seqs = sorted(pop.seq.unique())
        lines.append(f"- Video được lấy mẫu ({len(seqs)}): {', '.join(seqs)}")
    if len(ok):
        tab = pd.crosstab(ok.type, ok.label).reindex(columns=list(LABELS), fill_value=0)
        lines += ["", "Theo loại lỗi:", "", tab.to_markdown()]
    if len(bad):
        lines += ["", "Chưa gán nhãn: " + ", ".join(str(x) for x in bad.event_id)]
    lines += ["", "Lưu ý: mẫu nhỏ nên khoảng tin cậy rộng; tỷ lệ chỉ áp dụng cho các video có trong quần thể lấy mẫu."]
    report = "\n".join(lines)
    with open(os.path.join(os.path.dirname(os.path.abspath(labels_csv)), "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    return report


# ---------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="split folder with one sub-folder per sequence (gt/gt.txt, img1/)")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--out", default="outputs/spot_check")
    ap.add_argument("--n", type=int, default=30, help="number of switches to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pre", type=float, default=1.0, help="seconds shown before the switch")
    ap.add_argument("--post", type=float, default=1.0, help="seconds shown after the switch")
    ap.add_argument("--speed", type=float, default=0.5, help="playback speed of the clips (0.5 = half speed)")
    ap.add_argument("--remote", action="store_true", help="read missing frames from the Hugging Face zips")
    ap.add_argument("--zip-dir", default=None, help="folder with downloaded <sequence>.zip files")
    ap.add_argument("--summarize", metavar="LABELS_CSV", default=None, help="summarise a labelled labels.csv")
    args = ap.parse_args()
    if args.summarize:
        print(summarize(args.summarize))
        return
    if not args.data:
        ap.error("--data is required (or use --summarize)")
    cfg = load_config(args.config)
    seqs = discover_sequences(args.data)
    if args.seqs:
        seqs = [s for s in seqs if s.name in set(args.seqs)]
    if not seqs:
        sys.exit("no sequences")
    os.makedirs(args.out, exist_ok=True)
    tracked, pops = {}, []
    for s in seqs:
        tr = track_oracle(s, cfg)
        sw = classify_switches(tr["cm"], tr["tracks"])
        avail = has_frames(s, args.remote, args.zip_dir)
        if len(sw):
            sw = sw.assign(seq=s.name, in_population=avail)
            dens = pd.Series(tr["gt"][:, F].astype(int)).value_counts()
            sw["density"] = sw.frame.map(dens).fillna(0).astype(int)
            pops.append(sw)
        tracked[s.name] = (s, tr)
        print(f"[track] {s.name}: {len(sw)} switches, frames: {'yes' if avail else 'no'}")
    if not pops:
        sys.exit("no identity switches found")
    allsw = pd.concat(pops, ignore_index=True)[["seq", "frame", "gt_id", "prev_trk", "new_trk", "type", "density",
                                                 "in_population"]]
    allsw.to_csv(os.path.join(args.out, "switches_all.csv"), index=False)
    pop = allsw[allsw.in_population]
    left_out = sorted(set(allsw.seq) - set(pop.seq))
    if left_out:
        print(f"[note] {len(left_out)} sequences have no frame source and are not sampled: {', '.join(left_out)}. "
              "Use --remote (Hugging Face) or --zip-dir to include them.")
    if len(pop) == 0:
        sys.exit("no switch in a sequence with frames; add --remote or --zip-dir")
    sample = sample_switches(pop, args.n, args.seed)
    sample.insert(0, "event_id", [f"{i + 1:02d}" for i in range(len(sample))])
    print(f"[sample] {len(sample)} of {len(pop)} switches; by type: {sample.type.value_counts().to_dict()}")
    rows = []
    for seq_name, evs in sample.groupby("seq", sort=True):
        s, tr = tracked[seq_name]
        src, how = frame_source(s, args.remote, args.zip_dir)
        size = (s.width or 1920, s.height or 1080)
        try:
            for ev in evs.to_dict("records"):
                hints = gt_hints(tr["gt"], tr["gt_vis"], int(ev["gt_id"]), int(ev["frame"]))
                files = render_event(ev, tr, src, size, s.fps, args.out, args.pre, args.post, args.speed)
                rows.append({**ev, **hints, **files, "frames_from": how, "speed": args.speed, "label": "", "note": ""})
                print(f"  #{ev['event_id']} {seq_name} frame {ev['frame']} ({ev['type']}) -> {files['clip']}")
        finally:
            if src is not None:
                src.close()
    out = pd.DataFrame(rows).sort_values("event_id").reset_index(drop=True)
    cols = ["event_id", "seq", "frame", "gt_id", "prev_trk", "new_trk", "type", "density", "gt_gap_before",
            "gt_iou_min", "gt_vis", "n_gt_overlap", "gt_dup", "clip", "sheet", "clip_from", "clip_to", "frames_from", "speed",
            "label", "note"]
    out = out[cols]
    out.to_csv(os.path.join(args.out, "labels.csv"), index=False)
    page = write_index(args.out, out)
    print(f"\nlabels: {os.path.join(args.out, 'labels.csv')}\npage:   {page}\n"
          f"Next: open index.html, label every event, export labels.csv into {args.out}, then run\n"
          f"  python scripts/spot_check_switches.py --summarize {os.path.join(args.out, 'labels.csv')}")


if __name__ == "__main__":
    main()
