"""Plumbing test for Sam3Backend with a fake SAM 3 predictor (no GPU / torch needed)."""
import contextlib
import os
import sys
import types

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rad.io.mot import Sequence  # noqa: E402
from rad.sam.backends import Sam3Backend  # noqa: E402


class T(np.ndarray):
    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


class FakePredictor:
    """Mimics Sam3TrackerPredictor: box in relative coords, masks at video resolution."""

    def init_state(self, video_path, offload_video_to_cpu, async_loading_frames):
        frames = sorted(os.listdir(video_path))
        assert frames[0].startswith("00000"), frames[:3]
        im = Image.open(os.path.join(video_path, frames[0]))
        return {"n": len(frames), "size": im.size}

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, box):
        assert box.shape == (1, 4) and box.max() <= 1.0
        inference_state["box"] = box[0]

    def propagate_in_video(self, state, start_frame_idx, max_frame_num_to_track, reverse,
                           propagate_preflight, tqdm_disable):
        w, h = state["size"]
        x1, y1, x2, y2 = np.round(state["box"] * np.array([w, h, w, h])).astype(int)
        for i in range(min(state["n"], max_frame_num_to_track)):
            m = np.zeros((1, 1, h, w), dtype=float) - 1
            m[0, 0, y1:y2, x1 + i:x2 + i] = 1.0
            yield i, [1], None, m.view(T), np.array([[2.0]]).view(T)


def test_sam3_backend_plumbing(tmp_path):
    img = tmp_path / "S" / "img1"
    img.mkdir(parents=True)
    for f in range(1, 21):
        Image.new("RGB", (160, 120)).save(img / f"{f:06d}.jpg")
    seq = Sequence(name="S", root=str(tmp_path / "S"), img_dir=str(img), width=160, height=120, length=20)
    be = Sam3Backend(seq, tmp_root=str(tmp_path))
    fake_torch = types.SimpleNamespace(
        inference_mode=contextlib.nullcontext,
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    be._torch = fake_torch
    be.predictor = FakePredictor()
    be._autocast = contextlib.nullcontext
    out = list(be.propagate(5, np.array([20, 30, 60, 70]), 10))
    assert [f for f, _, _ in out] == list(range(5, 15))
    f0, m0, s0 = out[0]
    assert s0 == 2.0 and m0.area() == 40 * 40
    assert np.allclose(m0.bbox(), [20, 30, 60, 70])
    assert not [p for p in os.listdir(tmp_path) if p.startswith("rad_clip_")]  # temp clip removed
