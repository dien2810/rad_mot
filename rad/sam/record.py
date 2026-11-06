"""Wrappers around a VOS backend, used by the demo (scripts/make_demo.py).

RecordingBackend  keeps what each propagate() call produced (mask outline or box per
                  frame), so the demo can draw the SAM masks of every window.
CachingBackend    replays a propagate() call with the same start frame and seed box
                  instead of running the model again. Two dispatch policies often open
                  the same window (same track, same seed); with SAM 3 on a T4 this saves
                  minutes. Masks are stored bit-packed (1080p: ~0.26 MB per frame).

Both keep the backend interface: propagate(start_frame, box_xyxy, max_len) yields
(frame, mask, score), and other attributes are forwarded to the wrapped backend.
"""
from __future__ import annotations

import time
from typing import Iterator, Optional

import numpy as np

from rad.sam.masks import ArrayMask


def mask_shape(mask, simplify_px: float = 1.5):
    """('box', xyxy) for box masks, ('poly', [Nx2 int arrays]) for pixel masks, ('empty', None)."""
    if not hasattr(mask, "mask"):
        box = np.asarray(mask.bbox(), float)
        return ("empty", None) if mask.area() <= 0 else ("box", box)
    m = np.asarray(mask.mask)
    if not m.any():
        return ("empty", None)
    import cv2

    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = [cv2.approxPolyDP(c, simplify_px, True).reshape(-1, 2).astype(np.int32)
             for c in contours if cv2.contourArea(c) >= 4]
    return ("poly", polys) if polys else ("box", mask.bbox())


class RecordingBackend:
    """Records, per propagate() call, {frame: (shape, score)} in `calls` (in call order)."""

    def __init__(self, inner, simplify_px: float = 1.5, verbose: bool = False, label: str = ""):
        self.inner = inner
        self.calls: list[dict] = []
        self.simplify_px = simplify_px
        self.verbose = verbose
        self.label = label

    def __getattr__(self, k):  # forward name, stats(), frames_processed, ...
        return getattr(self.inner, k)

    def propagate(self, start_frame: int, box_xyxy, max_len: int) -> Iterator[tuple[int, object, float]]:
        rec = dict(start=int(start_frame), box=np.asarray(box_xyxy, float).copy(), frames={})
        self.calls.append(rec)
        t0 = time.perf_counter()
        gen = self.inner.propagate(start_frame, box_xyxy, max_len)
        try:
            for f, mask, score in gen:
                rec["frames"][int(f)] = (mask_shape(mask, self.simplify_px), float(score))
                yield f, mask, score
        finally:
            gen.close()
            if self.verbose:
                print(f"  [{self.label}] window {len(self.calls)}: seed frame {start_frame}, "
                      f"{len(rec['frames'])} frames, {time.perf_counter() - t0:.1f}s", flush=True)


def attach_shapes(windows, calls: list[dict]) -> dict[int, dict]:
    """Map recorded calls to windows: every window with a seed frame made exactly one call,
    in the order of `windows` (rad.sam.smp.run_smp). Returns {window index: call record}."""
    out: dict[int, dict] = {}
    k = 0
    for i, w in enumerate(windows):
        if w.seed_frame is None:
            continue
        if k < len(calls) and calls[k]["start"] == int(w.seed_frame):
            out[i] = calls[k]
            k += 1
        else:  # should not happen; fall back to the first unused call with this start frame
            used = {id(c) for c in out.values()}
            hit = next((c for c in calls if c["start"] == int(w.seed_frame) and id(c) not in used), None)
            if hit is not None:
                out[i] = hit
    return out


class _Packed:
    __slots__ = ("bits", "shape")

    def __init__(self, mask: np.ndarray):
        self.bits = np.packbits(mask, axis=None)
        self.shape = mask.shape

    def unpack(self) -> ArrayMask:
        n = int(np.prod(self.shape))
        return ArrayMask(np.unpackbits(self.bits, count=n).reshape(self.shape).astype(bool))


class CachingBackend:
    """Replays identical calls (same start frame, same seed box rounded to 0.01 px).

    A replay serves the cached frames first; if the caller wants more frames than were
    produced before (the earlier caller stopped early) the model is re-run from the seed
    and the extra frames are appended. Frame outputs do not depend on max_len (forward
    propagation only), so the result equals an uncached run.
    """

    def __init__(self, inner, max_frames: Optional[int] = None):
        self.inner = inner
        self.cache: dict[tuple, dict] = {}
        self.max_frames = max_frames  # stop caching new frames beyond this many (memory guard)
        self.stored = 0
        self.hits = 0
        self.misses = 0
        self.frames_replayed = 0

    def __getattr__(self, k):
        return getattr(self.inner, k)

    @staticmethod
    def _key(start_frame: int, box_xyxy) -> tuple:
        return (int(start_frame), tuple(np.round(np.asarray(box_xyxy, float), 2).tolist()))

    def _store(self, mask):
        if isinstance(mask, ArrayMask):
            return _Packed(mask.mask)
        return mask  # BoxMask and other light objects

    @staticmethod
    def _load(item):
        return item.unpack() if isinstance(item, _Packed) else item

    def propagate(self, start_frame: int, box_xyxy, max_len: int) -> Iterator[tuple[int, object, float]]:
        key = self._key(start_frame, box_xyxy)
        ent = self.cache.get(key)
        served = 0
        if ent is not None:
            self.hits += 1
            for f, item, score in ent["items"]:
                if served >= max_len:
                    return
                served += 1
                self.frames_replayed += 1
                yield f, self._load(item), score
            if served >= max_len or (ent["complete_len"] is not None and max_len <= ent["complete_len"]):
                return
        else:
            self.misses += 1
            ent = dict(items=[], complete_len=None)
            self.cache[key] = ent
        # (re-)run the model; skip the frames already served from the cache
        gen = self.inner.propagate(start_frame, box_xyxy, max_len)
        try:
            for i, (f, mask, score) in enumerate(gen):
                if i < served:
                    continue
                if i == len(ent["items"]) and (self.max_frames is None or self.stored < self.max_frames):
                    ent["items"].append((f, self._store(mask), score))
                    self.stored += 1
                served += 1
                yield f, mask, score
            if served == len(ent["items"]):  # the model ran out of frames and everything is cached
                ent["complete_len"] = max_len
        finally:
            gen.close()
