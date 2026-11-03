"""Load configs/*.yaml into the dataclass parameter objects."""
from __future__ import annotations

from dataclasses import fields
from typing import Optional

import yaml

from rad.dispatch.candidates import CandidateParams
from rad.sam.smp import SMPParams
from rad.tracker.bytetrack import ByteTrackParams


def _build(cls, d: Optional[dict]):
    d = d or {}
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise KeyError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    if "image_size" in d:
        d = {**d, "image_size": tuple(d["image_size"])}
    return cls(**d)


def load_config(path: Optional[str]) -> dict:
    raw = {}
    if path:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
    return dict(
        bytetrack=_build(ByteTrackParams, raw.get("bytetrack")),
        smp=_build(SMPParams, raw.get("smp")),
        candidates=_build(CandidateParams, raw.get("candidates")),
        oracle_dets=raw.get("oracle_dets", {}),
        oracle_backend=raw.get("oracle_backend", {}),
        sam3_backend=raw.get("sam3_backend", {}) or {},
        sweep=raw.get("sweep", {}),
        gt_ignore_classes=list(raw.get("gt_ignore_classes", []) or []),
        det_ignore_ioa=float(raw.get("det_ignore_ioa", 0.5)),
        raw=raw,
    )
