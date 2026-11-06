"""Video object segmentation (VOS) backends used inside SMP windows.

Each backend implements
    propagate(start_frame, box_xyxy, max_len) -> iterator of (frame, mask, score)
starting at `start_frame` (1-based MOT frame) with a box prompt.

OracleBackend  - follows the GT object under the seed box (CPU, no images). Useful to
                 develop/debug the dispatch policy and as an upper bound for VOS quality.
                 Optional drift/dropout simulate VOS failures.
Sam3Backend    - SAM 3 tracker (SAM 2 style API, box prompt), needs a CUDA GPU, the
                 `sam3` package and access to the gated checkpoint on Hugging Face.
"""
from __future__ import annotations

import contextlib
import gc
import os
import shutil
import tempfile
import time
from typing import Iterator, Optional

import numpy as np

from rad.boxes import iou_matrix
from rad.io.mot import ID, V, X, Y, W, H, Sequence, group_by_frame
from rad.sam.masks import ArrayMask, BoxMask


def rss_gb() -> float:
    """Resident memory of this process in GB (psutil if present, else /proc; nan if unknown)."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024 / 1e9
        except OSError:
            pass
    return float("nan")


class OracleBackend:
    name = "oracle"

    def __init__(self, gt: np.ndarray, iou_min: float = 0.3, vis_min: float = 0.3,
                 drift_prob: float = 0.0, drift_iou: float = 0.3, seed: int = 0):
        self.by_frame = group_by_frame(gt)
        self.iou_min = iou_min
        self.vis_min = vis_min
        self.drift_prob = drift_prob
        self.drift_iou = drift_iou
        self.rng = np.random.default_rng(seed)
        self.frames_processed = 0
        self.calls = 0

    @staticmethod
    def _xyxy(rows: np.ndarray) -> np.ndarray:
        return np.stack([rows[:, X], rows[:, Y], rows[:, X] + rows[:, W], rows[:, Y] + rows[:, H]], 1)

    def propagate(self, start_frame: int, box_xyxy: np.ndarray, max_len: int
                  ) -> Iterator[tuple[int, object, float]]:
        self.calls += 1
        rows = self.by_frame.get(start_frame)
        if rows is None or len(rows) == 0:
            return
        ious = iou_matrix(np.asarray(box_xyxy)[None], self._xyxy(rows))[0]
        k = int(np.argmax(ious))
        if ious[k] < self.iou_min:
            return
        gid = int(rows[k, ID])
        for f in range(start_frame, start_frame + max_len):
            rows = self.by_frame.get(f)
            self.frames_processed += 1
            if rows is None:
                yield f, BoxMask([0, 0, 0, 0]), -1.0
                continue
            sel = np.flatnonzero(rows[:, ID].astype(int) == gid)
            if len(sel) == 0 or rows[sel[0], V] < self.vis_min:
                yield f, BoxMask([0, 0, 0, 0]), -1.0
                continue
            box = self._xyxy(rows[sel])[0]
            if self.drift_prob > 0:
                others = np.flatnonzero(rows[:, ID].astype(int) != gid)
                if len(others):
                    ov = iou_matrix(box[None], self._xyxy(rows[others]))[0]
                    cand = others[ov >= self.drift_iou]
                    if len(cand) and self.rng.random() < self.drift_prob:
                        gid = int(rows[self.rng.choice(cand), ID])
                        box = self._xyxy(rows[rows[:, ID].astype(int) == gid])[0]
            yield f, BoxMask(box), 1.0


class Sam3Backend:
    """SAM 3 tracker with a box prompt on a temporary JPEG sub-clip.

    Frames of the window are symlinked (copied on Windows) as 00000.jpg, 00001.jpg, ...
    so that SAM 3 only loads the window, not the whole sequence.

    precision:
      "auto" - keep SAM 3's own setting: Sam3TrackerPredictor enters a global bfloat16
               autocast when it is built. Native on Ampere+ (L4, A100, RTX 30xx+); on a
               T4 / P100 bfloat16 is emulated, so it runs but may be slow.
      "fp16" - experimental for T4: leave SAM 3's bfloat16 autocast and use float16
               autocast instead (T4 tensor cores support fp16). Check accuracy with
               scripts/smoke_sam3.py before using it for results.
    """
    name = "sam3"
    _cache: dict = {}  # checkpoint -> predictor, shared across sequences

    def __init__(self, seq: Sequence, checkpoint_path: Optional[str] = None,
                 offload_video_to_cpu: bool = True, tmp_root: Optional[str] = None,
                 score_thresh: float = 0.0, precision: str = "auto", verbose: bool = True):
        if precision not in ("auto", "fp16"):
            raise ValueError("precision must be 'auto' or 'fp16'")
        self.precision = precision
        self.seq = seq
        self.paths = seq.frame_paths()
        if not self.paths:
            raise FileNotFoundError(
                f"{seq.name}: no image frames found (expected {seq.root}/img1). "
                "Extract frames first: python scripts/extract_frames.py")
        self.checkpoint_path = checkpoint_path
        self.offload = offload_video_to_cpu
        self.tmp_root = tmp_root or tempfile.gettempdir()
        self.score_thresh = score_thresh
        self.verbose = verbose  # one line per window: frames, time, RAM, GPU memory (find OOM causes)
        self.predictor = None
        self.frames_processed = 0
        self.calls = 0
        self.gpu_seconds = 0.0
        self.peak_mem_gb = 0.0
        self.width = seq.width
        self.height = seq.height

    def _build(self):
        import torch

        self._torch = torch
        key = self.checkpoint_path or "hf"
        if key not in Sam3Backend._cache:
            from sam3.model_builder import build_sam3_video_model

            kw = {}
            if self.checkpoint_path:
                kw = dict(checkpoint_path=self.checkpoint_path, load_from_HF=False)
            model = build_sam3_video_model(**kw)
            predictor = model.tracker
            predictor.backbone = model.detector.backbone
            Sam3Backend._cache[key] = predictor
        self.predictor = Sam3Backend._cache[key]
        if self.precision == "fp16" and getattr(self.predictor, "_rad_bf16_off", False) is False:
            ctx = getattr(self.predictor, "bf16_context", None)
            if ctx is not None:
                ctx.__exit__(None, None, None)  # undo the global bf16 autocast entered in __init__
            self.predictor._rad_bf16_off = True
        if self.width is None or self.height is None:
            from PIL import Image

            with Image.open(self.paths[0]) as im:
                self.width, self.height = im.size

    def _autocast(self):
        torch = self._torch
        if self.precision == "fp16" and torch.cuda.is_available():
            return torch.autocast("cuda", dtype=torch.float16)
        # "auto": SAM 3 already runs under its own global bf16 autocast. Note that
        # torch.cuda.is_bf16_supported() returns True on a T4 too (emulation), so it is
        # not used to decide anything here.
        return contextlib.nullcontext()

    @staticmethod
    def gpu_info() -> dict:
        import torch

        if not torch.cuda.is_available():
            return dict(gpu=None)
        p = torch.cuda.get_device_properties(0)
        return dict(gpu=p.name, vram_gb=round(p.total_memory / 1e9, 1), capability=f"{p.major}.{p.minor}",
                    native_bf16=p.major >= 8)

    def _make_clip(self, start_frame: int, n: int) -> str:
        clip = tempfile.mkdtemp(prefix="rad_clip_", dir=self.tmp_root)
        for i in range(n):
            src = self.paths[start_frame - 1 + i]
            dst = os.path.join(clip, f"{i:05d}{os.path.splitext(src)[1].lower()}")
            try:
                os.symlink(os.path.abspath(src), dst)
            except OSError:  # Windows without symlink rights
                shutil.copyfile(src, dst)
        return clip

    def propagate(self, start_frame: int, box_xyxy: np.ndarray, max_len: int
                  ) -> Iterator[tuple[int, object, float]]:
        if self.predictor is None:
            self._build()
        torch = self._torch
        n = min(max_len, len(self.paths) - start_frame + 1)
        if n <= 0:
            return
        self.calls += 1
        clip = self._make_clip(start_frame, n)
        state = None
        done = 0
        t0 = time.perf_counter()
        try:
            with torch.inference_mode(), self._autocast():
                state = self.predictor.init_state(video_path=clip, offload_video_to_cpu=self.offload,
                                                  async_loading_frames=True)
                x1, y1, x2, y2 = [float(v) for v in box_xyxy]
                rel = np.array([[x1 / self.width, y1 / self.height, x2 / self.width, y2 / self.height]],
                               dtype=np.float32)
                self.predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1, box=rel)
                for fi, obj_ids, _low, video_res, scores in self.predictor.propagate_in_video(
                        state, start_frame_idx=0, max_frame_num_to_track=n, reverse=False,
                        propagate_preflight=True, tqdm_disable=True):
                    m = video_res[0]
                    while m.ndim > 2:
                        m = m[0]
                    mask = (m > 0).cpu().numpy()
                    score = float(scores[0].flatten()[0]) if scores is not None else 0.0
                    self.frames_processed += 1
                    done += 1
                    if score < self.score_thresh:
                        mask = np.zeros_like(mask)
                    yield start_frame + int(fi), ArrayMask(mask), score
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self.peak_mem_gb = max(self.peak_mem_gb, torch.cuda.max_memory_allocated() / 1e9)
            dt = time.perf_counter() - t0
            self.gpu_seconds += dt
            del state
            shutil.rmtree(clip, ignore_errors=True)  # also stops SAM 3's frame-loading thread
            gc.collect()  # the inference state holds the window's frames (~1.8 GB for 150 frames at 1008 px)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.verbose:
                gpu = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                print(f"[sam3] {self.seq.name} window {self.calls}: seed frame {start_frame}, {done} frames, "
                      f"{dt:.0f}s, RAM {rss_gb():.1f} GB, GPU {gpu:.1f} GB (peak {self.peak_mem_gb:.1f})",
                      flush=True)

    def stats(self) -> dict:
        return dict(calls=self.calls, frames=self.frames_processed, seconds=self.gpu_seconds,
                    sec_per_frame=self.gpu_seconds / max(self.frames_processed, 1),
                    peak_mem_gb=self.peak_mem_gb)
