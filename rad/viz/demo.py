"""Drawing for the demo video (scripts/make_demo.py).

Layout "grid" (default, 16:9):   ByteTrack | SMP
                                 RAD       | cost / ID-switch chart
Layout "single": one panel at full size (for slides).

Boxes and masks are drawn with OpenCV; all text (Vietnamese) is drawn in one PIL pass
per frame with DejaVu Sans (ships with matplotlib).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from rad.io.mot import ID, X, Y, W, H, group_by_frame

# colours (RGB)
INK = (31, 35, 40)
MUTED = (101, 109, 118)
GRID = (216, 222, 228)
PAPER = (250, 251, 252)
BAR = (22, 27, 34)
WINDOW = (0, 186, 255)      # SAM mask of a window opened by the policy
WITNESS = (178, 126, 255)   # SAM mask of a witness window
SEED_FAIL = (160, 166, 173)
IDSW = (236, 72, 58)
FIXED = (52, 199, 105)
POLICY = {"base": (110, 118, 129), "smp": (47, 111, 189), "rad": (212, 118, 44)}


def bgr(rgb) -> tuple:
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def id_color(i: int) -> tuple:
    """Stable, well separated colour per track id (RGB)."""
    h = (int(i) * 0.618033988749895) % 1.0
    hsv = np.uint8([[[int(h * 180), 170, 235]]])
    r, g, b = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return int(r), int(g), int(b)


@lru_cache(maxsize=32)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    try:
        from matplotlib import font_manager

        path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans",
                                                                 weight="bold" if bold else "normal"))
        return ImageFont.truetype(path, size)
    except Exception:  # pragma: no cover
        return ImageFont.load_default()


@dataclass
class Text:
    x: float
    y: float
    s: str
    size: int = 14
    fill: tuple = (255, 255, 255)
    bg: Optional[tuple] = None
    bold: bool = False
    anchor: str = "la"   # PIL anchor: l/m/r + a(scender)/m(iddle)/d(escender)


def draw_texts(canvas_bgr: np.ndarray, texts: list[Text]) -> np.ndarray:
    if not texts:
        return canvas_bgr
    im = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    for t in texts:
        f = font(int(t.size), t.bold)
        if t.bg is not None:
            x1, y1, x2, y2 = d.textbbox((t.x, t.y), t.s, font=f, anchor=t.anchor)
            pad = max(2, int(t.size * 0.22))
            d.rectangle((x1 - pad, y1 - pad, x2 + pad, y2 + pad), fill=t.bg)
        d.text((t.x, t.y), t.s, font=f, fill=t.fill, anchor=t.anchor)
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


class View:
    """Maps image coordinates of a crop (x0, y0, w, h) into a panel of size (pw, ph)."""

    def __init__(self, crop: tuple, pw: int, ph: int):
        self.x0, self.y0, self.w, self.h = [float(v) for v in crop]
        self.pw, self.ph = pw, ph
        self.s = min(pw / self.w, ph / self.h)
        self.ox = (pw - self.w * self.s) / 2
        self.oy = (ph - self.h * self.s) / 2

    def pt(self, x, y):
        return (np.asarray(x, float) - self.x0) * self.s + self.ox, (np.asarray(y, float) - self.y0) * self.s + self.oy

    def box(self, xyxy) -> np.ndarray:
        x1, y1 = self.pt(xyxy[0], xyxy[1])
        x2, y2 = self.pt(xyxy[2], xyxy[3])
        return np.array([x1, y1, x2, y2])

    def poly(self, pts: np.ndarray) -> np.ndarray:
        x, y = self.pt(pts[:, 0], pts[:, 1])
        return np.stack([x, y], 1).round().astype(np.int32)

    def image(self, img: Optional[np.ndarray]) -> np.ndarray:
        out = np.full((self.ph, self.pw, 3), 40, np.uint8)
        if img is None:
            return out
        x0, y0 = int(round(self.x0)), int(round(self.y0))
        x1, y1 = int(round(self.x0 + self.w)), int(round(self.y0 + self.h))
        sub = img[max(y0, 0):y1, max(x0, 0):x1]
        if sub.size == 0:
            return out
        tw, th = max(int(round(sub.shape[1] * self.s)), 1), max(int(round(sub.shape[0] * self.s)), 1)
        interp = cv2.INTER_AREA if self.s < 1 else cv2.INTER_LINEAR
        sub = cv2.resize(sub, (tw, th), interpolation=interp)
        ox, oy = int(round(self.ox)), int(round(self.oy))
        h, w = min(th, self.ph - oy), min(tw, self.pw - ox)
        out[oy:oy + h, ox:ox + w] = sub[:h, :w]
        return out


def _dashed_rect(img, b, color, thick=2, dash=8):
    x1, y1, x2, y2 = [int(round(v)) for v in b]
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        n = max(int(np.hypot(bx - ax, by - ay) // dash), 1)
        for k in range(0, n, 2):
            p = (int(ax + (bx - ax) * k / n), int(ay + (by - ay) * k / n))
            q = (int(ax + (bx - ax) * min(k + 1, n) / n), int(ay + (by - ay) * min(k + 1, n) / n))
            cv2.line(img, p, q, color, thick, cv2.LINE_AA)


def _clip_texts(items: list[Text], x0: float, y0: float, w: float, h: float) -> list[Text]:
    """Keep labels inside the visible image area of a panel (drop those far outside, shift the rest)."""
    out = []
    for t in items:
        f = font(int(t.size), t.bold)
        tw, th = f.getlength(t.s), t.size * 1.15
        if t.y < y0 - 2 * th or t.y > y0 + h + 2 * th or t.x < x0 - 0.5 * w or t.x > x0 + 1.5 * w:
            continue
        a_h, a_v = t.anchor[0], t.anchor[1]
        left = t.x - (tw if a_h == "r" else tw / 2 if a_h == "m" else 0)
        top = t.y - (th if a_v in "ds" else th / 2 if a_v == "m" else 0)
        dx = max(x0 + 3 - left, 0) + min(x0 + w - 3 - (left + tw), 0)
        dy = max(y0 + 3 - top, 0) + min(y0 + h - 3 - (top + th), 0)
        t.x, t.y = t.x + dx, t.y + dy
        out.append(t)
    return out


def nice_ceil(v: float) -> float:
    if v <= 0:
        return 1.0
    e = 10 ** np.floor(np.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * e:
            return float(m * e)
    return float(10 * e)


def _blend_rows(img, y0, y1, rgb, alpha):
    y0, y1 = max(int(y0), 0), min(int(y1), img.shape[0])
    if y1 > y0:
        img[y0:y1] = (img[y0:y1] * (1 - alpha) + np.array(bgr(rgb)) * alpha).astype(np.uint8)


@dataclass
class PanelData:
    key: str                          # base / smp / rad
    title: str
    subtitle: str
    tracks: np.ndarray                # tracker output shown in this panel (after correction)
    windows: list = field(default_factory=list)
    shapes: dict = field(default_factory=dict)       # window index -> recorded VOS call
    scores: dict = field(default_factory=dict)       # window index -> trigger score text
    switches: pd.DataFrame = field(default_factory=pd.DataFrame)   # CLEAR switches in the clip
    vos_name: str = "SAM"
    s0: int = 1
    s1: int = 1

    def __post_init__(self):
        self.by_frame = group_by_frame(self.tracks)
        n = self.s1 - self.s0 + 1
        self.cum_idsw = np.zeros(n, int)
        self.cum_windows = np.zeros(n, int)
        self.cum_ran = np.zeros(n, int)
        self.cum_vos = np.zeros(n, int)
        if len(self.switches):
            for f in self.switches.frame.astype(int):
                if self.s0 <= f <= self.s1:
                    self.cum_idsw[f - self.s0:] += 1
        for w in self.windows:
            f = int(w.entry_frame)
            if self.s0 <= f <= self.s1:
                k = f - self.s0
                self.cum_windows[k:] += 1
                if w.seed_frame is not None:
                    self.cum_ran[k:] += 1
                    self.cum_vos[k:] += int(w.frames)

    def at(self, f: int, arr: np.ndarray) -> int:
        return int(arr[min(max(f - self.s0, 0), len(arr) - 1)])


def _track_box(by_frame, f, tid):
    rows = by_frame.get(f)
    if rows is None:
        return None
    sel = rows[rows[:, ID].astype(int) == int(tid)]
    if len(sel) == 0:
        return None
    r = sel[0]
    return np.array([r[X], r[Y], r[X] + r[W], r[Y] + r[H]])


def draw_panel(canvas: np.ndarray, texts: list[Text], ox: int, oy: int, pw: int, ph: int,
               img: Optional[np.ndarray], f: int, pd_: PanelData, view: View, label_all: bool = True,
               focus_ids: set = frozenset()) -> None:
    k = pw / 960.0                                      # font / line scale
    fs = lambda v: max(int(round(v * k)), 9)  # noqa: E731
    lw = max(int(round(2 * k)), 1)
    hh, fh = int(48 * k), int(30 * k)                   # header / footer height
    outer = texts
    texts = []                                          # labels inside the image area (clipped below)
    panel = view.image(img)
    # 1. masks of active windows (one translucent layer)
    active = [(i, w) for i, w in enumerate(pd_.windows)
              if w.seed_frame is not None and w.end_frame is not None and w.seed_frame <= f <= w.end_frame]
    layer = panel.copy()
    outlines = []
    for i, w in active:
        rec = pd_.shapes.get(i)
        if rec is None or f not in rec["frames"]:
            continue
        (kind, geom), _score = rec["frames"][f]
        col = bgr(WITNESS if w.witness_of is not None else WINDOW)
        if kind == "poly":
            polys = [view.poly(p) for p in geom]
            cv2.fillPoly(layer, polys, col)
            outlines.append(("poly", polys, col, w))
        elif kind == "box":
            b = view.box(geom)
            outlines.append(("box", b, col, w))
    if outlines:
        cv2.addWeighted(layer, 0.38, panel, 0.62, 0, panel)
    # 2. tracker boxes
    rows = pd_.by_frame.get(f)
    n_rows = 0 if rows is None else len(rows)
    if rows is not None:
        for r in rows:
            tid = int(r[ID])
            b = view.box([r[X], r[Y], r[X] + r[W], r[Y] + r[H]])
            if b[2] < 0 or b[3] < 0 or b[0] > pw or b[1] > ph:
                continue
            c = id_color(tid)
            focus = tid in focus_ids
            cv2.rectangle(panel, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), bgr(c),
                          lw + (1 if focus else 0), cv2.LINE_AA)
            if label_all or focus or (b[3] - b[1]) >= 36 * k:
                lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
                texts.append(Text(ox + b[0], oy + b[1] - 2, str(tid), fs(12), (0, 0, 0) if lum > 150 else (255, 255, 255),
                                  bg=c, bold=True, anchor="ld"))
    # 3. window outlines + labels
    for kind, geom, col, w in outlines:
        if kind == "poly":
            cv2.polylines(panel, geom, True, col, lw + 1, cv2.LINE_AA)
            pts = np.concatenate(geom)
            tx, ty = pts[:, 0].min(), pts[:, 1].max()
        else:
            _dashed_rect(panel, geom, col, lw + 1)
            tx, ty = geom[0], geom[3]
        tag = pd_.vos_name + (" (witness)" if w.witness_of is not None else "")
        texts.append(Text(ox + tx, oy + ty + 3, tag, fs(12), (255, 255, 255),
                          bg=WITNESS if w.witness_of is not None else WINDOW, bold=True, anchor="la"))
    # 4. events
    for i, w in enumerate(pd_.windows):
        e = int(w.entry_frame)
        if w.witness_of is None and e <= f <= e + 14:
            b = _track_box(pd_.by_frame, f, w.track_id)
            if b is not None:
                vb = view.box(b)
                if w.seed_frame is None:
                    _dashed_rect(panel, vb, bgr(SEED_FAIL), lw + 1)
                    texts.append(Text(ox + vb[0], oy + vb[3] + 3, "không có khung mồi", fs(11), INK,
                                      bg=SEED_FAIL, anchor="la"))
                else:
                    g = 6 * k
                    cv2.rectangle(panel, (int(vb[0] - g), int(vb[1] - g)), (int(vb[2] + g), int(vb[3] + g)),
                                  bgr(WINDOW), lw + 1, cv2.LINE_AA)
                    sc = pd_.scores.get(i)
                    texts.append(Text(ox + vb[2] + g + 2, oy + vb[1] - g, "gọi " + pd_.vos_name + (f" · {sc}" if sc else ""),
                                      fs(12), (255, 255, 255), bg=WINDOW, bold=True, anchor="la"))
        if w.applied and w.swap_frame is not None and int(w.swap_frame) <= f <= int(w.swap_frame) + 36:
            a, b_id = int(w.track_id), int(w.settled_id)
            for tid in (a, b_id):
                bb = _track_box(pd_.by_frame, f, tid)
                if bb is None:
                    continue
                vb = view.box(bb)
                cv2.rectangle(panel, (int(vb[0]), int(vb[1])), (int(vb[2]), int(vb[3])), bgr(FIXED), lw + 2, cv2.LINE_AA)
            bb = _track_box(pd_.by_frame, f, a) if _track_box(pd_.by_frame, f, a) is not None else _track_box(pd_.by_frame, f, b_id)
            if bb is not None:
                vb = view.box(bb)
                texts.append(Text(ox + vb[0], oy + vb[3] + 3, f"sửa ID {a}↔{b_id}", fs(13), (255, 255, 255),
                                  bg=FIXED, bold=True, anchor="la"))
    if len(pd_.switches):
        recent = pd_.switches[(pd_.switches.frame <= f) & (pd_.switches.frame >= f - 24)]
        for s in recent.itertuples(index=False):
            bb = _track_box(pd_.by_frame, f, int(s.new_trk))
            if bb is None:
                continue
            vb = view.box(bb)
            cv2.rectangle(panel, (int(vb[0]), int(vb[1])), (int(vb[2]), int(vb[3])), bgr(IDSW), lw + 2, cv2.LINE_AA)
            texts.append(Text(ox + vb[0], oy + vb[3] + 3, f"đổi ID {int(s.prev_trk)}→{int(s.new_trk)}", fs(13),
                              (255, 255, 255), bg=IDSW, bold=True, anchor="la"))
    outer.extend(_clip_texts(texts, ox, oy + hh, pw, ph - hh - fh))
    texts = outer
    # 5. header and footer bars
    _blend_rows(panel, 0, hh, BAR, 0.72)
    _blend_rows(panel, ph - fh, ph, BAR, 0.72)
    cv2.rectangle(panel, (0, 0), (int(5 * k), hh), bgr(POLICY.get(pd_.key, MUTED)), -1)
    canvas[oy:oy + ph, ox:ox + pw] = panel
    texts.append(Text(ox + 14 * k, oy + 7 * k, pd_.title, fs(19), (255, 255, 255), bold=True))
    texts.append(Text(ox + 14 * k, oy + 30 * k, pd_.subtitle, fs(12), (200, 207, 214)))
    if any(True for _ in active):
        texts.append(Text(ox + pw - 12 * k, oy + 12 * k, f"● {pd_.vos_name} đang chạy: {len(active)}", fs(13),
                          WINDOW, bold=True, anchor="ra"))
    idsw = pd_.at(f, pd_.cum_idsw)
    if pd_.key == "base":
        foot = f"Lỗi đổi ID: {idsw}      không gọi {pd_.vos_name}"
    else:
        foot = (f"Lỗi đổi ID: {idsw}      Cửa sổ: {pd_.at(f, pd_.cum_windows)} (chạy {pd_.at(f, pd_.cum_ran)})"
                f"      Khung {pd_.vos_name}: {pd_.at(f, pd_.cum_vos)}")
    texts.append(Text(ox + 14 * k, oy + ph - fh / 2, foot, fs(14), (255, 255, 255), bold=True, anchor="lm"))
    texts.append(Text(ox + pw - 12 * k, oy + ph - fh / 2, f"khung {f}", fs(12), (200, 207, 214), anchor="rm"))
    if n_rows == 0 and img is None:
        texts.append(Text(ox + pw / 2, oy + ph / 2, "không có ảnh", fs(16), (230, 230, 230), anchor="mm"))


def draw_chart(canvas: np.ndarray, texts: list[Text], ox: int, oy: int, pw: int, ph: int, f: int,
               panels: list[PanelData], caption: str, has_gt: bool = True) -> None:
    k = pw / 960.0
    fs = lambda v: max(int(round(v * k)), 9)  # noqa: E731
    canvas[oy:oy + ph, ox:ox + pw] = np.array(bgr(PAPER), np.uint8)
    s0, s1 = panels[0].s0, panels[0].s1
    left, right = int(70 * k), int(130 * k)
    vos = panels[-1].vos_name
    plots = [(f"Khung {vos} đã xử lý (cộng dồn)", "cum_vos", [p for p in panels if p.key != "base"]),
             ("Lỗi đổi ID trong đoạn (cộng dồn)", "cum_idsw", panels) if has_gt else
             (f"Cửa sổ {vos} đã chạy (cộng dồn; không có GT nên không đếm lỗi)", "cum_ran",
              [p for p in panels if p.key != "base"])]
    top = int(50 * k)
    ph_plot = int(150 * k)
    gap = int(62 * k)
    texts.append(Text(ox + 20 * k, oy + 14 * k, "Chi phí và độ chính xác theo thời gian", fs(18), INK, bold=True))
    for j, (title, attr, ps) in enumerate(plots):
        y0 = oy + top + j * (ph_plot + gap) + int(28 * k)
        y1 = y0 + ph_plot
        x0, x1 = ox + left, ox + pw - right
        texts.append(Text(ox + 20 * k, y0 - 24 * k, title, fs(14), INK, bold=True))
        ymax = nice_ceil(max([int(getattr(p, attr)[-1]) for p in ps] + [1]) * 1.05)
        for t in np.linspace(0, ymax, 3):
            yy = int(y1 - (y1 - y0) * t / ymax)
            cv2.line(canvas, (x0, yy), (x1, yy), bgr(GRID), 1, cv2.LINE_AA)
            texts.append(Text(x0 - 8 * k, yy, f"{t:,.0f}".replace(",", "."), fs(11), MUTED, anchor="rm"))
        xs = np.linspace(x0, x1, s1 - s0 + 1)
        cur = min(max(f - s0, 0), s1 - s0)
        label_y = []
        for p in ps:
            v = getattr(p, attr)
            ys = y1 - (y1 - y0) * v / ymax
            pts = np.stack([xs, ys], 1).round().astype(np.int32)
            col = bgr(POLICY[p.key])
            faint = tuple(int(c * 0.25 + 255 * 0.75) for c in col)
            cv2.polylines(canvas, [pts], False, faint, max(int(2 * k), 1), cv2.LINE_AA)
            cv2.polylines(canvas, [pts[:cur + 1]], False, col, max(int(3 * k), 2), cv2.LINE_AA)
            cv2.circle(canvas, tuple(pts[cur]), max(int(4 * k), 2), col, -1, cv2.LINE_AA)
            ly = float(pts[-1][1])
            for prev in label_y:  # avoid overlapping end labels
                if abs(ly - prev) < 15 * k:
                    ly = prev + 15 * k
            label_y.append(ly)
            name = {"base": "ByteTrack", "smp": "SMP", "rad": "RAD"}[p.key]
            texts.append(Text(x1 + 8 * k, ly, f"{name}: {int(v[cur])}", fs(12), POLICY[p.key], bold=True,
                              anchor="lm"))
        cx = int(xs[cur])
        cv2.line(canvas, (cx, y0), (cx, y1), bgr(MUTED), 1, cv2.LINE_AA)
        texts.append(Text(x0, y1 + 4 * k, f"khung {s0}", fs(10), MUTED, anchor="la"))
        texts.append(Text(x1, y1 + 4 * k, f"khung {s1}", fs(10), MUTED, anchor="ra"))
    # legend of the overlays
    ly = oy + ph - int(78 * k)
    items = [(f"mask {vos}" if vos != "VOS" else "vùng VOS", WINDOW, True), ("witness", WITNESS, True), ("không có khung mồi", SEED_FAIL, False),
             ("đổi ID (lỗi)", IDSW, False), ("sửa ID", FIXED, False)]
    x = ox + 20 * k
    for name, col, fill in items:
        sq = int(12 * k)
        if fill:
            cv2.rectangle(canvas, (int(x), int(ly)), (int(x + sq), int(ly + sq)), bgr(col), -1)
        cv2.rectangle(canvas, (int(x), int(ly)), (int(x + sq), int(ly + sq)), bgr(col), max(int(2 * k), 1))
        texts.append(Text(x + sq + 6 * k, ly + sq / 2, name, fs(12), INK, anchor="lm"))
        x += sq + 6 * k + font(fs(12)).getlength(name) + 20 * k
    for j, line in enumerate(caption.split("\n")[:2]):
        texts.append(Text(ox + 20 * k, oy + ph - int((36 - 17 * j) * k), line, fs(11), MUTED, anchor="lm"))


class VideoWriter:
    """H.264 MP4 through ffmpeg (plays in browsers / Colab); falls back to OpenCV mp4v."""

    def __init__(self, path: str, size: tuple, fps: float, crf: int = 23):
        self.path = path
        self.size = size
        self.proc = None
        self.cv = None
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{size[0]}x{size[1]}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264",
                   "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", str(crf), "-movflags", "+faststart", path]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame_bgr: np.ndarray) -> None:
        if self.proc is not None:
            self.proc.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())
        else:
            self.cv.write(frame_bgr)

    def close(self) -> None:
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
        elif self.cv is not None:
            self.cv.release()
