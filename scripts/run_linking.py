#!/usr/bin/env python
"""Method 3 (learned tracklet linking) and method 4 (VOS check of uncertain links) on top of ByteTrack.

For each video: ByteTrack on the given detections (YOLO by default), every (A ended, C born)
pair near in time and space, P(same object) from a gradient-boosted classifier, then
  bytetrack     - the tracker alone
  link_upper    - every true pair merged (a perfect classifier): the ceiling of linking
  link          - method 3: merge pairs with p >= tau_high (no SAM)
  link_verify   - method 4: also verify pairs with tau_low <= p < tau_high with the VOS backend
                  (--backend oracle: masks from GT, an upper bound; --backend sam3: real SAM 3)

No label of a test video is used to choose anything for it. The classifier is trained out of
fold by video (GroupKFold, 5 folds), and tau_high / tau_low are chosen inside each training
fold from inner out-of-fold scores:
  tau_high maximises (correct merges - lam * wrong merges) on the training videos;
  tau_low  is the lowest threshold whose band [tau_low, tau_high) still holds at least
           --band-precision true pairs on the training videos (a verification is worth a call).
At inference the model and both thresholds are fixed; nothing needs labels.

Results per video go to <out>/per_seq/<seq>.csv as soon as it is done (videos already there are
skipped, so an interrupted SAM 3 run resumes), the scored pairs to <out>/pairs/<seq>.csv, and the
totals with bootstrap intervals over videos to <out>/summary.md.

Examples:
    python scripts/run_linking.py --data datasets/FastTracker/train --dets dir --det-dir outputs/dets_yolo \
        --backend oracle --out outputs/link_yolo/oracle
    python scripts/run_linking.py ... --backend sam3 --precision fp16 --frames-on-gpu --max-verify 20 \
        --out outputs/link_yolo/sam3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rad.config import load_config  # noqa: E402
from rad.dispatch.model import fit_full, train_oof  # noqa: E402
from rad.eval.idsw import clear_match, idf1  # noqa: E402
from rad.io.mot import discover_sequences  # noqa: E402
from rad.link import LINK_FEATURES, LinkParams, apply_links, assign, label_links, link_candidates, verify_pair  # noqa: E402
from rad.pilot.analysis import classify_switches  # noqa: E402
from rad.pipeline import load_detections, load_gt, make_backend, prepare  # noqa: E402

GRID = np.round(np.arange(0.05, 1.0, 0.05), 2)


def metrics(gt: np.ndarray, trk: np.ndarray) -> dict:
    cm = clear_match(gt, trk)
    out = dict(idsw=cm.num_switches, mota=cm.mota, fp=cm.num_fp, fn=cm.num_fn)
    out["idf1"] = idf1(gt, trk)["idf1"]
    types = classify_switches(cm, trk)["type"].value_counts() if cm.num_switches else pd.Series(dtype=int)
    for t in ("fragment", "swap", "takeover"):
        out[t] = int(types.get(t, 0))
    return out


def choose_taus(train: pd.DataFrame, score: str, lam: float, band_precision: float) -> tuple[float, float]:
    best, tau_high = -np.inf, 1.0
    for tau in GRID:
        tp = fp = 0
        for _, g in train.groupby("seq"):
            sel = assign(g, score, tau)
            tp += int(sel.y.sum())
            fp += int(len(sel) - sel.y.sum())
        obj = tp - lam * fp
        if obj > best or (obj == best and tau > tau_high):
            best, tau_high = obj, float(tau)
    tau_low = tau_high
    for t in sorted([t for t in GRID if t < tau_high], reverse=True):
        band = train[(train[score] >= t) & (train[score] < tau_high)]
        if len(band) and band.y.mean() >= band_precision:
            tau_low = float(t)
    return tau_high, tau_low


def nested_scores(lc: pd.DataFrame, lam: float, band_precision: float, kind: str = "hgb") -> tuple[pd.Series, pd.DataFrame]:
    """Out-of-fold P(same object) for every pair, and per-video thresholds chosen without its labels."""
    seqs = lc.seq.to_numpy()
    n_groups = len(np.unique(seqs))
    p = pd.Series(np.nan, index=lc.index)
    taus = []
    gkf = GroupKFold(n_splits=min(5, n_groups))
    for fold, (tr, te) in enumerate(gkf.split(lc, lc.y, seqs)):
        train, test = lc.iloc[tr].copy(), lc.iloc[te]
        inner, _ = train_oof(train, "y", LINK_FEATURES, kind=kind, n_splits=min(4, train.seq.nunique()))
        train["p_in"] = inner.fillna(0.0)
        th, tl = choose_taus(train, "p_in", lam, band_precision)
        if train.y.sum() and train.y.sum() < len(train):
            model = fit_full(train, "y", LINK_FEATURES, kind=kind)
            p.iloc[te] = model.predict_proba(test[LINK_FEATURES].astype(float).to_numpy())[:, 1]
        else:
            p.iloc[te] = 0.0
        for s in np.unique(test.seq):
            taus.append(dict(seq=s, fold=fold, tau_high=th, tau_low=tl))
    return p, pd.DataFrame(taus).set_index("seq")


def bootstrap_sum(x: np.ndarray, n_boot: int = 20000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), size=(n_boot, len(x)))].sum(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def summarise(out: str) -> str:
    files = sorted(glob.glob(os.path.join(out, "per_seq", "*.csv")))
    if not files:
        return "no results yet"
    per = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    per.to_csv(os.path.join(out, "all_per_seq.csv"), index=False)
    g = per.groupby("policy", sort=False)
    cols = ["idsw", "fragment", "swap", "takeover", "merges", "merges_wrong", "verify_calls", "verify_confirmed",
            "verify_wrong", "vos_frames", "sec_vos"]
    summ = g[cols].sum()
    summ["videos"] = g.seq.nunique()
    summ["idf1_mean"] = g.idf1.mean()
    summ["mota_mean"] = g.mota.mean()
    summ = summ.reindex([p for p in ("bytetrack", "link_upper", "link", "link_verify") if p in summ.index])
    summ.to_csv(os.path.join(out, "summary.csv"))
    lines = ["# Linking: summary", "", summ.round(3).to_string(), ""]
    piv = per.pivot_table(index="seq", columns="policy", values="idsw")
    for pol in ("link_upper", "link", "link_verify"):
        if pol in piv and "bytetrack" in piv:
            d = (piv.bytetrack - piv[pol]).dropna().to_numpy()
            lo, hi = bootstrap_sum(d)
            lines.append(f"IDSW reduction vs ByteTrack, {pol}: {d.sum():.0f} (95% bootstrap over {len(d)} videos: "
                         f"{lo:.0f} to {hi:.0f}); better on {(d > 0).sum()}, worse on {(d < 0).sum()} videos")
    txt = "\n".join(lines)
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write(txt + "\n")
    return txt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=False)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    ap.add_argument("--dets", choices=["dir", "det", "oracle"], default="dir")
    ap.add_argument("--det-dir", default="outputs/dets_yolo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["none", "oracle", "sam3"], default="oracle")
    ap.add_argument("--precision", choices=["auto", "fp16"], default="fp16")
    ap.add_argument("--frames-on-gpu", action="store_true")
    ap.add_argument("--max-verify", type=int, default=0, help="most verifications per video, highest p first (0 = all)")
    ap.add_argument("--lam", type=float, default=2.0, help="cost of a wrong merge relative to a correct one")
    ap.add_argument("--band-precision", type=float, default=0.2)
    ap.add_argument("--max-gap-s", type=float, default=3.0)
    ap.add_argument("--radius", type=float, default=5.0)
    ap.add_argument("--run", nargs="*", default=None, help="videos to evaluate, in this order (default: all)")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "per_seq"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "pairs"), exist_ok=True)
    if args.summary_only:
        print(summarise(args.out))
        return
    if not args.data:
        sys.exit("--data is required")
    cfg = load_config(args.config)
    lp = LinkParams(max_gap_s=args.max_gap_s, radius=args.radius)
    ign = dict(ignore_classes=cfg["gt_ignore_classes"], ignore_ioa=cfg["det_ignore_ioa"])

    t0 = time.perf_counter()
    data, parts = {}, []
    for s in discover_sequences(args.data):
        _, gt_vis = load_gt(s, **ign)
        det = load_detections(s, args.dets, args.det_dir, cfg["oracle_dets"], gt_vis=gt_vis, **ign)
        sd = prepare(s, det, cfg["bytetrack"], cfg["candidates"], **ign)
        sd.cands = None  # RAD's match candidates are not used here
        t = time.perf_counter()
        lc = link_candidates(sd.tracks, sd.size, s.fps or 30.0, lp)
        lc["seq"] = s.name
        lc["y"] = label_links(lc, sd.cm.matches, lp.label_k)
        data[s.name] = (sd, time.perf_counter() - t)
        parts.append(lc)
        print(f"[prepare] {s.name}: {len(sd.tracks):,} track boxes, {len(lc):,} pairs, {int(lc.y.sum())} true",
              flush=True)
    lc_all = pd.concat(parts, ignore_index=True)
    print(f"[prepare] {len(data)} videos, {len(lc_all):,} pairs, {int(lc_all.y.sum())} true "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)

    t = time.perf_counter()
    lc_all["p"], taus = nested_scores(lc_all, args.lam, args.band_precision)
    ok = lc_all.p.notna()
    mm = dict(n=int(len(lc_all)), positives=int(lc_all.y.sum()), seconds=round(time.perf_counter() - t, 1))
    if ok.sum() and lc_all.y[ok].nunique() == 2:
        mm.update(roc_auc=float(roc_auc_score(lc_all.y[ok], lc_all.p[ok])),
                  pr_auc=float(average_precision_score(lc_all.y[ok], lc_all.p[ok])))
    mm["taus"] = taus.reset_index().to_dict("records")
    with open(os.path.join(args.out, "model_metrics.json"), "w") as fh:
        json.dump(mm, fh, indent=1)
    print("[model]", {k: v for k, v in mm.items() if k != "taus"}, flush=True)
    print(taus.to_string(), flush=True)

    kw = {}
    if args.backend == "oracle":
        kw = dict(cfg["oracle_backend"])
    elif args.backend == "sam3":
        kw = dict(cfg["sam3_backend"])
        kw["precision"] = args.precision
        if args.frames_on_gpu:
            kw["offload_video_to_cpu"] = False
    order = args.run or list(data)
    done = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(args.out, "per_seq", "*.csv"))}
    for name in order:
        if name in done:
            print(f"[skip] {name}: already in per_seq/")
            continue
        sd, sec_cands = data[name]
        if args.backend == "sam3" and not sd.seq.frame_paths():
            print(f"[skip] {name}: no img1/ frames")
            continue
        lc = lc_all[lc_all.seq == name].copy()
        th, tl = float(taus.loc[name, "tau_high"]), float(taus.loc[name, "tau_low"])
        common = dict(seq=name, frames=sd.length, n_pairs=len(lc), n_true=int(lc.y.sum()), tau_high=th, tau_low=tl,
                      sec_link=sec_cands)
        rows = []
        zero = dict(merges=0, merges_wrong=0, verify_calls=0, verify_confirmed=0, verify_wrong=0, vos_frames=0, sec_vos=0.0)
        rows.append(dict(**common, policy="bytetrack", **metrics(sd.gt, sd.tracks), **zero))
        up = assign(lc.assign(yf=lc.y.astype(float)), "yf", 0.5)
        rows.append(dict(**common, policy="link_upper", **metrics(sd.gt, apply_links(sd.tracks, up)),
                         **{**zero, "merges": len(up)}))
        sel = assign(lc, "p", th)
        rows.append(dict(**common, policy="link", **metrics(sd.gt, apply_links(sd.tracks, sel)),
                         **{**zero, "merges": len(sel), "merges_wrong": int((sel.y == 0).sum())}))
        lc["decision"] = "none"
        lc.loc[lc.index.isin(sel.index), "decision"] = "merge"
        if args.backend != "none":
            band = lc[(lc.p >= tl) & (lc.p < th) & ~lc.a.isin(sel.a) & ~lc.c.isin(sel.c)].sort_values("p", ascending=False)
            if args.max_verify:
                band = band.head(args.max_verify)
            be = make_backend(args.backend, sd, **kw)
            infos, t = [], time.perf_counter()
            for r in band.itertuples():
                info = verify_pair(be, sd.tracks, int(r.a), int(r.c), lp)
                info["idx"] = r.Index
                infos.append(info)
            sec_vos = time.perf_counter() - t
            conf_idx = [i["idx"] for i in infos if i["confirmed"]]
            lc.loc[band.index, "decision"] = "verify_rejected"
            lc.loc[conf_idx, "decision"] = "verify_confirmed"
            extra = assign(lc.loc[conf_idx], "p", 0.0) if conf_idx else lc.iloc[:0]
            final = pd.concat([sel, extra])
            rows.append(dict(**common, policy="link_verify", **metrics(sd.gt, apply_links(sd.tracks, final)),
                             merges=len(final), merges_wrong=int((final.y == 0).sum()), verify_calls=len(infos),
                             verify_confirmed=len(conf_idx), verify_wrong=int((lc.loc[conf_idx].y == 0).sum()),
                             vos_frames=int(sum(i["frames"] for i in infos)), sec_vos=sec_vos))
            if infos:
                pd.DataFrame(infos).to_csv(os.path.join(args.out, "pairs", f"{name}_verify.csv"), index=False)
            del be
        lc.to_csv(os.path.join(args.out, "pairs", f"{name}.csv"), index=False)
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "per_seq", f"{name}.csv"), index=False)
        r = {x["policy"]: x for x in rows}
        msg = f"[done] {name}: IDSW ByteTrack {r['bytetrack']['idsw']}, upper {r['link_upper']['idsw']}, " \
              f"link {r['link']['idsw']} ({r['link']['merges']} merges, {r['link']['merges_wrong']} wrong)"
        if "link_verify" in r:
            v = r["link_verify"]
            msg += f", link+verify {v['idsw']} ({v['verify_calls']} checks, {v['verify_confirmed']} confirmed, " \
                   f"{v['verify_wrong']} wrong, {v['sec_vos'] / 60:.1f} min)"
        print(msg, flush=True)
        summarise(args.out)
    print(summarise(args.out))


if __name__ == "__main__":
    main()
