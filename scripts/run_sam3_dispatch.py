#!/usr/bin/env python
"""SMP and RAD with real SAM 3 on every video: one video at a time, resumable, timed.

Same setting as the oracle sweep of run_dispatch.py (oracle detections, out-of-fold risk
model, RAD budget b = 1, i.e. as many triggers as SMP at tau = 0.05). What it adds:

  * each policy gets its own SAM 3 cache, so a policy's time is what it would cost alone
    (run_dispatch.py shares one cache across policies, so the second policy looks cheaper);
  * wall-clock time per stage: ByteTrack, candidates + features, risk scoring, SAM 3 windows,
    and frames per second of the whole chain (detector excluded: detections are oracle);
  * results of a video go to <out>/per_seq/<seq>.csv as soon as it is done, and videos already
    there are skipped, so a dead Colab session only loses the video in progress;
  * out-of-fold risk scores are cached in <out>/risk_scores.csv.gz (trained once per output
    folder, reused after a restart).

Image frames (img1/) are needed only for the videos in --run; the risk model is trained on
the GT of every video under --data.

Example (Colab, T4):
    python scripts/run_sam3_dispatch.py --data /content/datasets/FastTracker/train \
        --precision fp16 --run task_night_occlusion task_day_right_turn_occlusion \
        --out /content/drive/MyDrive/.../outputs/dispatch_sam3_all
    python scripts/run_sam3_dispatch.py --out .../dispatch_sam3_all --summary-only \
        --old .../outputs/dispatch_sam3/sweep_per_sequence.csv
CPU dry run of the whole chain (oracle VOS instead of SAM 3, no frames needed):
    python scripts/run_sam3_dispatch.py --data ... --backend oracle --out /tmp/dry
"""
from __future__ import annotations

import argparse
import gc
import glob
import os
import pickle
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rad.config import load_config  # noqa: E402
from rad.dispatch import policy as pol  # noqa: E402
from rad.dispatch.candidates import FEATURES, build_candidates  # noqa: E402
from rad.dispatch.model import fit_full  # noqa: E402
from rad.eval.idsw import evaluate  # noqa: E402
from rad.io.mot import discover_sequences  # noqa: E402
from rad.pipeline import load_detections, load_gt, make_backend, prepare, train_risk  # noqa: E402
from rad.sam.record import CachingBackend  # noqa: E402
from rad.sam.smp import SMPParams, run_smp  # noqa: E402
from rad.tracker.bytetrack import ByteTrackParams, run_bytetrack  # noqa: E402

SCORE = "p_y_risk"
ORDER = {"learned": 0, "smp": 1, "smp+birth": 2}  # RAD first: nothing in any cache yet


def release_memory() -> None:
    """Return freed heap pages to the OS (glibc keeps them otherwise, so RSS creeps up per window)."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


class CountingBackend:
    """Counts the frames the VOS model really produces (cache replays are not counted).

    Also trims the heap after every window: a SAM 3 window holds up to 150 frames
    (~1.8 GB at 1008 px when they live in CPU RAM), and Colab kills the process
    without a Python error once RAM passes ~12.7 GB.
    """

    def __init__(self, inner):
        self.inner = inner
        self.frames = 0

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def propagate(self, start_frame, box_xyxy, max_len):
        gen = self.inner.propagate(start_frame, box_xyxy, max_len)
        try:
            for item in gen:
                self.frames += 1
                yield item
        finally:
            gen.close()
            release_memory()


def parse_policies(items: list[str]) -> list[tuple[str, float]]:
    out = []
    for it in items:
        name, _, val = it.partition(":")
        if name not in ORDER:
            sys.exit(f"unknown policy {name!r} (use {', '.join(ORDER)})")
        out.append((name, float(val or (1.0 if name == "learned" else 0.05))))
    return sorted(out, key=lambda x: (ORDER[x[0]], x[1]))


def load_or_train_scores(data, out: str) -> dict:
    """Out-of-fold scores, cached per output folder; returns the model metrics."""
    path = os.path.join(out, "risk_scores.csv.gz")
    n = {sd.name: len(sd.cands) for sd in data}
    if os.path.exists(path):
        sc = pd.read_csv(path)
        got = sc.groupby("seq").size().to_dict()
        if got == n:
            for sd in data:
                sd.cands[SCORE] = sc.loc[sc.seq == sd.name, SCORE].to_numpy()
            print(f"[model] reused out-of-fold scores from {path}")
            return dict(reused=True)
        print("[model] cached scores do not match the candidates (different data or code): retraining")
    t = time.perf_counter()
    metrics = train_risk(data, "y_risk")
    metrics["train_seconds"] = round(time.perf_counter() - t, 1)
    pd.concat([pd.DataFrame({"seq": sd.name, SCORE: sd.cands[SCORE].to_numpy()}) for sd in data]
              ).to_csv(path, index=False)
    print("[model]", {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
    return metrics


def timing_model(data, out: str):
    """One model on all videos, used only to time risk scoring (scores come from the OOF models)."""
    path = os.path.join(out, "timing_model.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    allc = pd.concat([sd.cands for sd in data], ignore_index=True)
    m = fit_full(allc, "y_risk")
    with open(path, "wb") as fh:
        pickle.dump(m, fh)
    return m


def time_cheap_stages(sd, cfg, model) -> dict:
    """Re-run ByteTrack and candidate building on this video, timed (outputs are identical)."""
    bt = ByteTrackParams(**{**asdict(cfg["bytetrack"]), "frame_rate": sd.seq.fps})
    t0 = time.perf_counter()
    tracks, log = run_bytetrack(sd.det, sd.length, bt)
    t1 = time.perf_counter()
    cands = build_candidates(log, tracks, cfg["candidates"])
    t2 = time.perf_counter()
    model.predict_proba(cands[list(FEATURES)].astype(float).to_numpy())
    t3 = time.perf_counter()
    if len(tracks) != len(sd.tracks) or len(cands) != len(sd.cands):
        print(f"[warn] {sd.name}: timed re-run differs from the prepared run (non-deterministic step?)")
    return dict(sec_bytetrack=t1 - t0, sec_candidates=t2 - t1, sec_score=t3 - t2)


def warm_up(sd, kw: dict) -> None:
    """Load SAM 3 and run a 3-frame window, so model loading is not billed to the first policy."""
    t = time.perf_counter()
    be = make_backend("sam3", sd, **{**kw, "verbose": False})
    f, _, x, y, w, h = sd.tracks[0, :6]  # frame, id, x, y, w, h of the first tracker box
    for _ in be.propagate(int(f), np.array([x, y, x + w, y + h]), 3):
        pass
    print(f"[sam3] model loaded and warmed up ({time.perf_counter() - t:.0f}s, not counted)", flush=True)


def run_video(sd, cfg, policies, backend: str, kw: dict, cheap: dict) -> list[dict]:
    p = SMPParams(**{**asdict(cfg["smp"]), "image_size": sd.size})
    fps_video = float(sd.seq.fps or 30)
    common = dict(seq=sd.name, frames=sd.length, video_fps=fps_video,
                  frames_on_gpu=int(backend == "sam3" and kw.get("offload_video_to_cpu", True) is False))
    base = evaluate(sd.gt, sd.tracks)
    rows = [dict(**common, policy="bytetrack", param=0.0, windows=0, vos_frames=0, model_frames=0, triggers=0,
                 idsw=base["idsw"], idf1=base["idf1"], mota=base["mota"], n_swap=0,
                 sec_bytetrack=cheap["sec_bytetrack"], sec_candidates=0.0, sec_score=0.0, sec_select=0.0,
                 sec_vos=0.0)]
    ref_n = len(pol.smp_baseline(sd.log, 0.05, p.gap_G))
    for name, param in policies:
        t0 = time.perf_counter()
        if name == "learned":
            trig = pol.budget(sd.cands.dropna(subset=[SCORE]), SCORE, int(round(param * max(ref_n, 1))))
        elif name == "smp":
            trig = pol.smp_baseline(sd.log, param, p.gap_G)
        else:
            trig = pol.smp_plus_births(sd.log, sd.cands, param, p.gap_G)
        t1 = time.perf_counter()
        counter = CountingBackend(make_backend(backend, sd, **kw))
        be = CachingBackend(counter, max_frames=1500) if backend == "sam3" else counter  # <= ~0.4 GB of masks
        corrected, wins, stats = run_smp(sd.tracks, sd.log, be, trig, sd.length, p)
        t2 = time.perf_counter()
        ev = evaluate(sd.gt, corrected)
        uses_cands = name != "smp"
        rows.append(dict(**common, policy=name, param=param, windows=stats["windows"],
                         vos_frames=stats["vos_frames"], model_frames=counter.frames, triggers=len(trig),
                         idsw=ev["idsw"], idf1=ev["idf1"], mota=ev["mota"], n_swap=stats.get("n_SWAP", 0),
                         sec_bytetrack=cheap["sec_bytetrack"],
                         sec_candidates=cheap["sec_candidates"] if uses_cands else 0.0,
                         sec_score=cheap["sec_score"] if name == "learned" else 0.0,
                         sec_select=t1 - t0, sec_vos=t2 - t1))
        del be, counter, corrected, wins
        release_memory()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        r = rows[-1]
        print(f"[{sd.name}] {name}:{param:g}  triggers {r['triggers']}  windows {r['windows']}  "
              f"VOS frames {r['vos_frames']}  IDSW {base['idsw']} -> {r['idsw']}  "
              f"VOS time {r['sec_vos'] / 60:.1f} min", flush=True)
    for r in rows:
        r["sec_total"] = r["sec_bytetrack"] + r["sec_candidates"] + r["sec_score"] + r["sec_select"] + r["sec_vos"]
        r["fps"] = r["frames"] / r["sec_total"] if r["sec_total"] > 0 else np.nan
        r["x_realtime"] = r["sec_total"] / (r["frames"] / r["video_fps"])
    return rows


def bootstrap(per: pd.DataFrame, a: str, b: str, col: str, n_boot: int = 20000, seed: int = 0):
    """Ratio sum(b)/sum(a) over videos, with a 95% interval from resampling videos."""
    piv = per.pivot_table(index="seq", columns="policy", values=col, aggfunc="sum")
    if a not in piv or b not in piv:
        return None
    piv = piv[[a, b]].dropna()
    if len(piv) < 2 or piv[a].sum() == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(piv), size=(n_boot, len(piv)))
    va, vb = piv[a].to_numpy()[idx].sum(1), piv[b].to_numpy()[idx].sum(1)
    ok = va > 0
    r = vb[ok] / va[ok]
    return dict(ratio=float(piv[b].sum() / piv[a].sum()), lo=float(np.percentile(r, 2.5)),
                hi=float(np.percentile(r, 97.5)), videos=int(len(piv)))


def summarise(out: str, old: str | None = None, verbose: bool = True) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(out, "per_seq", "*.csv")))
    if not files:
        sys.exit(f"no results in {out}/per_seq")
    new = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    new["source"] = "sam3_timed"
    per = new
    if old and os.path.exists(old):  # videos done earlier with run_dispatch.py (no per-policy timing)
        o = pd.read_csv(old)
        o = o[~o.seq.isin(new.seq.unique())].copy()
        if len(o):
            o["source"] = "sam3_old_untimed"
            per = pd.concat([new, o], ignore_index=True)
            if verbose:
                print(f"[summary] + {o.seq.nunique()} video(s) from {old} (counts only, no timing)")
    per.to_csv(os.path.join(out, "all_per_seq.csv"), index=False)
    key = ["policy", "param"]
    g = per.groupby(key)
    summ = g[["windows", "vos_frames", "triggers", "idsw"]].sum()
    summ["videos"] = g["seq"].nunique()
    summ["idf1_mean"] = g["idf1"].mean()
    timed = per[per.source == "sam3_timed"].groupby(key)
    summ["timed_videos"] = timed["seq"].nunique()
    summ["timed_frames"] = timed["frames"].sum()
    summ["timed_video_minutes"] = timed.apply(lambda d: (d.frames / d.video_fps).sum() / 60)
    summ["sec_vos"] = timed["sec_vos"].sum()
    summ["sec_total"] = timed["sec_total"].sum()
    summ["fps"] = summ["timed_frames"] / summ["sec_total"]
    summ["x_realtime"] = summ["sec_total"] / (summ["timed_video_minutes"] * 60)
    summ["sec_per_vos_frame"] = summ["sec_vos"] / timed["vos_frames"].sum().replace(0, np.nan)
    summ = summ.reset_index()
    summ.to_csv(os.path.join(out, "summary.csv"), index=False)
    lines = ["# SAM 3 dispatch: summary", "",
             summ.round(3).to_string(index=False), ""]
    for col, label in (("vos_frames", "VOS frames RAD/SMP"), ("sec_total", "time RAD/SMP (timed videos)")):
        src = per if col == "vos_frames" else per[per.source == "sam3_timed"]
        bs = bootstrap(src, "smp", "learned", col)
        if bs:
            lines.append(f"{label}: {bs['ratio']:.2f} (95% bootstrap over {bs['videos']} videos: "
                         f"{bs['lo']:.2f}-{bs['hi']:.2f})")
    piv = per.pivot_table(index="seq", columns="policy", values="idsw", aggfunc="sum")
    if {"smp", "learned", "bytetrack"} <= set(piv.columns):
        d = (piv["smp"] - piv["learned"]).dropna()
        rng = np.random.default_rng(1)
        boots = d.to_numpy()[rng.integers(0, len(d), size=(20000, len(d)))].sum(1)
        lines.append(f"IDSW SMP - RAD: {d.sum():.0f} (95% bootstrap: {np.percentile(boots, 2.5):.0f} to "
                     f"{np.percentile(boots, 97.5):.0f}); ByteTrack {piv['bytetrack'].sum():.0f}, "
                     f"SMP {piv['smp'].sum():.0f}, RAD {piv['learned'].sum():.0f}")
    txt = "\n".join(lines)
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write(txt + "\n")
    if verbose:
        print(txt)
    return summ


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="split folder with one sub-folder per video (gt/, seqinfo.ini, img1/)")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", nargs="*", default=None,
                    help="videos to run, in this order (default: every video with frames, or all for oracle)")
    ap.add_argument("--policies", nargs="*", default=["learned:1.0", "smp:0.05"],
                    help="policy:param, from learned (budget b), smp (tau), smp+birth (tau)")
    ap.add_argument("--backend", choices=["sam3", "oracle"], default="sam3")
    ap.add_argument("--precision", choices=["auto", "fp16"], default="fp16")
    ap.add_argument("--frames-on-gpu", action="store_true")
    ap.add_argument("--old", default=None, help="earlier sweep_per_sequence.csv to merge into the summary")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "per_seq"), exist_ok=True)
    if args.summary_only:
        summarise(args.out, args.old)
        return
    if not args.data:
        sys.exit("--data is required")
    cfg = load_config(args.config)
    policies = parse_policies(args.policies)
    seqs = discover_sequences(args.data)
    ign = dict(ignore_classes=cfg["gt_ignore_classes"], ignore_ioa=cfg["det_ignore_ioa"])
    t = time.perf_counter()
    data = []
    for s in seqs:
        _, gt_vis = load_gt(s, **ign)
        det = load_detections(s, "oracle", None, cfg["oracle_dets"], gt_vis=gt_vis, **ign)
        data.append(prepare(s, det, cfg["bytetrack"], cfg["candidates"], **ign))
    print(f"[prepare] {len(data)} videos, {sum(len(sd.cands) for sd in data):,} candidates "
          f"({time.perf_counter() - t:.0f}s)", flush=True)
    load_or_train_scores(data, args.out)
    tmodel = timing_model(data, args.out)
    by_name = {sd.name: sd for sd in data}
    order = args.run if args.run else [sd.name for sd in data]
    unknown = [s for s in order if s not in by_name]
    if unknown:
        sys.exit(f"unknown video(s): {unknown}; available: {sorted(by_name)}")
    done = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(args.out, "per_seq", "*.csv"))}
    # the other videos were only needed to train the risk model: free them before SAM 3 starts
    keep = {n for n in order if n not in done}
    for n in list(by_name):
        if n not in keep:
            del by_name[n]
    del data
    release_memory()
    kw = dict(cfg["oracle_backend"] if args.backend == "oracle" else cfg["sam3_backend"])
    if args.backend == "sam3":
        kw["precision"] = args.precision
        if args.frames_on_gpu:
            kw["offload_video_to_cpu"] = False
    todo = [n for n in order if n not in done and (args.backend == "oracle" or by_name[n].seq.frame_paths())]
    if args.backend == "sam3" and todo:
        warm_up(by_name[todo[0]], kw)
    for name in order:
        if name in done:
            print(f"[skip] {name}: already in {args.out}/per_seq")
            continue
        sd = by_name[name]
        if args.backend == "sam3" and not sd.seq.frame_paths():
            print(f"[skip] {name}: no img1/ frames")
            continue
        t = time.perf_counter()
        cheap = time_cheap_stages(sd, cfg, tmodel)
        rows = run_video(sd, cfg, policies, args.backend, kw, cheap)
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "per_seq", f"{name}.csv"), index=False)
        print(f"[done] {name} in {(time.perf_counter() - t) / 60:.1f} min -> per_seq/{name}.csv", flush=True)
        summarise(args.out, args.old, verbose=False)  # keep summary.csv/.md current if the session dies later
        del by_name[name], sd
        release_memory()
    summarise(args.out, args.old)


if __name__ == "__main__":
    main()
