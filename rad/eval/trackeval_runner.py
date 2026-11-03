"""HOTA / CLEAR / Identity via TrackEval (class-agnostic).

Install: pip install git+https://github.com/JonathonLuiten/TrackEval
TrackEval (2022 master) still uses np.float/np.int, removed in NumPy >= 1.24, so we
restore the aliases before importing it.

All GT boxes are evaluated as one class ("pedestrian" id 1 inside TrackEval). Use
`filter_classes` to evaluate a subset (e.g. motorbikes only): GT rows are filtered
by class and tracker rows by their majority class.
"""
from __future__ import annotations

import os
import tempfile
from typing import Iterable, Optional

import numpy as np

from rad.io.mot import C, write_mot_txt


def _patch_numpy() -> None:
    for name, val in (("float", float), ("int", int), ("bool", bool)):
        if not hasattr(np, name):
            setattr(np, name, val)


def run_trackeval(gt_by_seq: dict[str, np.ndarray], trk_by_seq: dict[str, np.ndarray],
                  seq_lengths: dict[str, int], tracker_name: str = "tracker",
                  filter_classes: Optional[Iterable[int]] = None,
                  workdir: Optional[str] = None) -> dict:
    """Return {'COMBINED': {...}, seq: {...}} with HOTA, DetA, AssA, MOTA, IDF1, IDSW."""
    _patch_numpy()
    import trackeval  # noqa: WPS433 (imported lazily on purpose)

    tmp = workdir or tempfile.mkdtemp(prefix="rad_trackeval_")
    gt_root = os.path.join(tmp, "gt")
    tr_root = os.path.join(tmp, "trackers")
    keep = set(int(c) for c in filter_classes) if filter_classes is not None else None
    for seq, gt in gt_by_seq.items():
        g = gt.copy()
        if keep is not None:
            g = g[np.isin(g[:, C].astype(int), list(keep))]
        g[:, 6] = 1.0   # conf -> considered
        g[:, 7] = 1.0   # class -> pedestrian (class-agnostic)
        g[:, 8] = 1.0
        write_mot_txt(os.path.join(gt_root, seq, "gt", "gt.txt"), g, tracker_format=False)
        t = trk_by_seq.get(seq, np.zeros((0, 9))).copy()
        if keep is not None and len(t):
            t = t[np.isin(t[:, C].astype(int), list(keep))]
        if len(t):
            t[:, C] = 1.0  # TrackEval's MOT dataset only accepts class 1 for trackers
        write_mot_txt(os.path.join(tr_root, tracker_name, "data", f"{seq}.txt"), t)

    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update(dict(USE_PARALLEL=False, PRINT_RESULTS=False, PRINT_CONFIG=False,
                         OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False, PLOT_CURVES=False,
                         TIME_PROGRESS=False, DISPLAY_LESS_PROGRESS=True, PRINT_ONLY_COMBINED=True))
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update(dict(GT_FOLDER=gt_root, TRACKERS_FOLDER=tr_root, TRACKERS_TO_EVAL=[tracker_name],
                       SKIP_SPLIT_FOL=True, SEQ_INFO=dict(seq_lengths), PRINT_CONFIG=False,
                       CLASSES_TO_EVAL=["pedestrian"], BENCHMARK="RAD", DO_PREPROC=True))
    evaluator = trackeval.Evaluator(eval_cfg)
    dataset = trackeval.datasets.MotChallenge2DBox(ds_cfg)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
               trackeval.metrics.Identity({"PRINT_CONFIG": False})]
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        res, _ = evaluator.evaluate([dataset], metrics)
    out = {}
    for seq, r in res["MotChallenge2DBox"][tracker_name].items():
        r = r["pedestrian"]
        out["COMBINED" if seq == "COMBINED_SEQ" else seq] = dict(
            HOTA=float(np.mean(r["HOTA"]["HOTA"])), DetA=float(np.mean(r["HOTA"]["DetA"])),
            AssA=float(np.mean(r["HOTA"]["AssA"])), MOTA=float(r["CLEAR"]["MOTA"]),
            IDF1=float(r["Identity"]["IDF1"]), IDSW=int(r["CLEAR"]["IDSW"]),
            Frag=int(r["CLEAR"]["Frag"]),
        )
    return out


def available() -> bool:
    try:
        _patch_numpy()
        import trackeval  # noqa: F401  (import check only)
        return True
    except Exception:
        return False
