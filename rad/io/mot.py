"""Read/write MOT-style text files and discover sequences.

Internal array layout (float64, shape (N, 9)), 1-based frames as in MOT:
    0 frame, 1 id, 2 x, 3 y, 4 w, 5 h, 6 score/conf, 7 cls, 8 vis

FastTracker benchmark gt.txt uses exactly this order:
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, 1.0
"""
from __future__ import annotations

import configparser
import glob
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

COLS = ["frame", "id", "x", "y", "w", "h", "score", "cls", "vis"]
F, ID, X, Y, W, H, S, C, V = range(9)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def load_mot_txt(path: str) -> np.ndarray:
    """Load a MOT-like txt file (comma or whitespace separated) into an (N, 9) array.

    Missing trailing columns are filled with score=1, cls=-1, vis=1.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, 9))
    rows = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            rows.append([float(p) for p in parts[:9]])
    width = max(len(r) for r in rows)
    if width < 6:
        raise ValueError(f"{path}: expected at least 6 columns, got {width}")
    out = np.zeros((len(rows), 9))
    out[:, S] = 1.0
    out[:, C] = -1.0
    out[:, V] = 1.0
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def write_mot_txt(path: str, arr: np.ndarray, tracker_format: bool = True) -> None:
    """Write an (N, >=8) array as MOT txt.

    tracker_format=True writes 10 columns: frame,id,x,y,w,h,score,cls,-1,-1
    (TrackEval reads the first 7; cls kept for per-class analysis).
    tracker_format=False writes GT style: frame,id,x,y,w,h,conf,cls,vis.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = np.asarray(arr)
    if len(arr):
        arr = arr[np.lexsort((arr[:, ID], arr[:, F]))]
    with open(path, "w") as fh:
        for r in arr:
            if tracker_format:
                fh.write(
                    f"{int(r[F])},{int(r[ID])},{r[X]:.2f},{r[Y]:.2f},{r[W]:.2f},{r[H]:.2f},"
                    f"{r[S]:.4f},{int(r[C])},-1,-1\n"
                )
            else:
                fh.write(
                    f"{int(r[F])},{int(r[ID])},{r[X]:.2f},{r[Y]:.2f},{r[W]:.2f},{r[H]:.2f},"
                    f"{r[S]:.4f},{int(r[C])},{r[V]:.4f}\n"
                )


def tlwh_to_xyxy(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=float)
    out = b.copy()
    out[..., 2] = b[..., 0] + b[..., 2]
    out[..., 3] = b[..., 1] + b[..., 3]
    return out


def xyxy_to_tlwh(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=float)
    out = b.copy()
    out[..., 2] = b[..., 2] - b[..., 0]
    out[..., 3] = b[..., 3] - b[..., 1]
    return out


def group_by_frame(arr: np.ndarray) -> dict[int, np.ndarray]:
    """Return {frame: rows} for an (N, 9) array."""
    out: dict[int, np.ndarray] = {}
    if len(arr) == 0:
        return out
    order = np.argsort(arr[:, F], kind="stable")
    arr = arr[order]
    frames = arr[:, F].astype(int)
    splits = np.flatnonzero(np.diff(frames)) + 1
    for chunk in np.split(arr, splits):
        out[int(chunk[0, F])] = chunk
    return out


@dataclass
class Sequence:
    name: str
    root: str
    gt_path: Optional[str] = None
    det_path: Optional[str] = None
    img_dir: Optional[str] = None
    video_path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: float = 30.0
    length: Optional[int] = None

    def load_gt(self) -> np.ndarray:
        return load_mot_txt(self.gt_path) if self.gt_path else np.zeros((0, 9))

    def frame_paths(self) -> list[str]:
        """Sorted image paths (index i -> frame i+1)."""
        if not self.img_dir:
            return []
        files = [p for p in os.listdir(self.img_dir) if p.lower().endswith(IMAGE_EXTS)]

        def key(p: str):
            stem = os.path.splitext(p)[0]
            digits = "".join(ch for ch in stem if ch.isdigit())
            return (int(digits) if digits else 0, p)

        files.sort(key=key)
        return [os.path.join(self.img_dir, p) for p in files]


def _read_seqinfo(path: str) -> dict:
    cp = configparser.ConfigParser()
    cp.read(path)
    if "Sequence" not in cp:
        return {}
    s = cp["Sequence"]
    return {k: s.get(k) for k in s}


def discover_sequences(split_dir: str) -> list[Sequence]:
    """Find sequences under split_dir (each subdir with gt/gt.txt, img1/ or a video)."""
    seqs: list[Sequence] = []
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(split_dir)
    for name in sorted(os.listdir(split_dir)):
        root = os.path.join(split_dir, name)
        if not os.path.isdir(root):
            continue
        gt = os.path.join(root, "gt", "gt.txt")
        det = os.path.join(root, "det", "det.txt")
        info = _read_seqinfo(os.path.join(root, "seqinfo.ini")) if os.path.exists(
            os.path.join(root, "seqinfo.ini")
        ) else {}
        img_dir = os.path.join(root, info.get("imdir", "img1"))
        if not os.path.isdir(img_dir):
            cands = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)
                     and any(f.lower().endswith(IMAGE_EXTS) for f in os.listdir(d)[:50])]
            img_dir = cands[0] if cands else None
        videos = [p for p in glob.glob(os.path.join(root, "*")) if p.lower().endswith(VIDEO_EXTS)]
        if not (os.path.exists(gt) or img_dir or videos):
            continue
        seq = Sequence(
            name=name,
            root=root,
            gt_path=gt if os.path.exists(gt) else None,
            det_path=det if os.path.exists(det) else None,
            img_dir=img_dir,
            video_path=videos[0] if videos else None,
            width=int(info["imwidth"]) if "imwidth" in info else None,
            height=int(info["imheight"]) if "imheight" in info else None,
            fps=float(info.get("framerate", 30) or 30),
            length=int(info["seqlength"]) if "seqlength" in info else None,
        )
        if seq.length is None and seq.gt_path:
            gt_arr = seq.load_gt()
            seq.length = int(gt_arr[:, F].max()) if len(gt_arr) else None
        if (seq.width is None or seq.height is None) and seq.img_dir:
            paths = seq.frame_paths()
            if paths:
                try:
                    from PIL import Image

                    with Image.open(paths[0]) as im:
                        seq.width, seq.height = im.size
                except Exception:  # pragma: no cover - PIL optional
                    pass
        seqs.append(seq)
    return seqs
