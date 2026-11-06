"""Recording / caching VOS wrappers used by the demo (run: python -m pytest -q)."""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rad.sam.backends import OracleBackend  # noqa: E402
from rad.sam.masks import ArrayMask  # noqa: E402
from rad.sam.record import CachingBackend, RecordingBackend, attach_shapes, mask_shape  # noqa: E402
from rad.sam.smp import SMPParams, run_smp, smp_triggers  # noqa: E402
from test_core import _crossing  # noqa: E402


class PixelOracle(OracleBackend):
    """Oracle that returns pixel masks (like SAM 3) and counts frames it really computed."""

    def __init__(self, gt, size=(720, 1280)):
        super().__init__(gt)
        self.size = size

    def propagate(self, start_frame, box_xyxy, max_len):
        for f, m, s in super().propagate(start_frame, box_xyxy, max_len):
            arr = np.zeros(self.size, bool)
            x1, y1, x2, y2 = np.round(m.bbox()).astype(int)
            arr[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = True
            yield f, ArrayMask(arr), s


def test_recording_maps_every_window_that_ran():
    gt, trk, log = _crossing()
    p = SMPParams(image_size=(1280, 720))
    rec = RecordingBackend(OracleBackend(gt))
    _, wins, stats = run_smp(trk, log, rec, smp_triggers(log, p), 100, p)
    shapes = attach_shapes(wins, rec.calls)
    ran = [i for i, w in enumerate(wins) if w.seed_frame is not None]
    assert sorted(shapes) == ran and len(rec.calls) == len(ran)
    for i in ran:
        assert len(shapes[i]["frames"]) == wins[i].frames
        kind, box = next(iter(shapes[i]["frames"].values()))[0]
        assert kind == "box" and box.shape == (4,)


def test_cache_gives_identical_results_and_saves_frames():
    gt, trk, log = _crossing()
    p = SMPParams(image_size=(1280, 720))
    trig = smp_triggers(log, p)
    ref, ref_w, _ = run_smp(trk, log, PixelOracle(gt), trig, 100, p)
    inner = PixelOracle(gt)
    cache = CachingBackend(inner)
    out1, w1, _ = run_smp(trk, log, cache, trig, 100, p)
    first = inner.frames_processed
    out2, w2, _ = run_smp(trk, log, RecordingBackend(cache), trig, 100, p)
    assert np.array_equal(out1, ref) and np.array_equal(out2, ref)
    assert [w.outcome for w in w2] == [w.outcome for w in ref_w]
    assert inner.frames_processed == first  # second run fully replayed
    assert cache.hits > 0 and cache.frames_replayed > 0


def test_cache_extends_an_entry_that_stopped_early():
    gt, _, _ = _crossing()
    inner = PixelOracle(gt)
    cache = CachingBackend(inner)
    box = np.array([104, 300, 144, 340])
    gen = cache.propagate(1, box, 40)
    short = [f for f, _, _ in itertools.islice(gen, 5)]
    gen.close()
    assert len(short) == 5
    full = [f for f, _, _ in cache.propagate(1, box, 40)]
    assert full == list(range(1, 41))
    again = [f for f, _, _ in cache.propagate(1, box, 40)]
    assert again == full


def test_mask_shape_polygon():
    arr = np.zeros((50, 80), bool)
    arr[10:30, 20:60] = True
    kind, polys = mask_shape(ArrayMask(arr))
    assert kind == "poly" and len(polys) == 1
    xs, ys = polys[0][:, 0], polys[0][:, 1]
    assert xs.min() == 20 and xs.max() == 59 and ys.min() == 10 and ys.max() == 29
