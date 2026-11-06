#!/usr/bin/env python
"""Window-by-window agreement between two VOS backends on the same tracker output.

Use it to check whether results measured with the cheap oracle VOS transfer to SAM 3:
both runs must use the same detections (e.g. both --dets dir with the YOLO folder), so
the triggers are identical and windows can be matched by (track_id, entry_frame, witness_of).

Example:
    python scripts/compare_backends.py --a $WORK/outputs/pilot_yolo --b $WORK/outputs/pilot_sam3 \
        --out $WORK/outputs/compare_oracle_sam3
Reads <a>/**/<seq>/smp_windows_oracle.csv and <b>/**/<seq>/smp_windows_sam3.csv (and the
per_sequence.csv files for IDSW). Writes per_sequence.csv, outcome_confusion.csv, report.md.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

KEY = ["track_id", "entry_frame", "witness_of"]
MODIFY = "SWAP"


def find(root: str, name: str) -> dict[str, str]:
    """{sequence: path} for every <root>/**/<seq>/<name>."""
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "**", name), recursive=True)):
        out.setdefault(os.path.basename(os.path.dirname(p)), p)
    return out


def per_seq_rows(root: str) -> pd.DataFrame:
    parts = [pd.read_csv(p) for p in glob.glob(os.path.join(root, "**", "per_sequence.csv"), recursive=True)]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).drop_duplicates("seq", keep="last").set_index("seq")


def compare(wa: pd.DataFrame, wb: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    for w in (wa, wb):
        w["witness_of"] = w["witness_of"].fillna(-1).astype(int)
    m = wa.merge(wb, on=KEY, how="outer", suffixes=("_a", "_b"), indicator=True)
    both = m[m["_merge"] == "both"]
    ran_a = both.frames_a.fillna(0) > 0
    ran_b = both.frames_b.fillna(0) > 0
    seeded = both.seed_frame_a.notna() & both.seed_frame_b.notna()
    s = dict(
        windows_a=len(wa), windows_b=len(wb), matched=len(both),
        only_a=int((m["_merge"] == "left_only").sum()), only_b=int((m["_merge"] == "right_only").sum()),
        seeded=int(seeded.sum()),
        ran_both=int((ran_a & ran_b).sum()),
        ran_b_not_a=int((~ran_a & ran_b & seeded).sum()),   # e.g. oracle cannot follow a box with no GT object
        ran_a_not_b=int((ran_a & ~ran_b & seeded).sum()),
        same_outcome_ran_both=int(((both.outcome_a == both.outcome_b) & ran_a & ran_b).sum()),
        same_swap_decision_ran_both=int((((both.outcome_a == MODIFY) == (both.outcome_b == MODIFY)) & ran_a & ran_b).sum()),
        swaps_a=int((wa.outcome == MODIFY).sum()), swaps_b=int((wb.outcome == MODIFY).sum()),
        applied_a=int(wa.applied.sum()) if "applied" in wa else np.nan,
        applied_b=int(wb.applied.sum()) if "applied" in wb else np.nan,
        vos_frames_a=int(wa.frames.sum()), vos_frames_b=int(wb.frames.sum()),
    )
    conf = pd.crosstab(both[seeded].outcome_a, both[seeded].outcome_b)
    return s, conf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="run folder of backend A (e.g. pilot_yolo, oracle VOS)")
    ap.add_argument("--b", required=True, help="run folder of backend B (e.g. pilot_sam3)")
    ap.add_argument("--a-name", default="oracle")
    ap.add_argument("--b-name", default="sam3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    fa = find(args.a, f"smp_windows_{args.a_name}.csv")
    fb = find(args.b, f"smp_windows_{args.b_name}.csv")
    seqs = sorted(set(fa) & set(fb))
    if not seqs:
        raise SystemExit(f"no sequence has both {args.a_name} and {args.b_name} windows "
                         f"(A: {sorted(fa)}, B: {sorted(fb)})")
    psa, psb = per_seq_rows(args.a), per_seq_rows(args.b)
    rows, confs = [], []
    for s in seqs:
        st, conf = compare(pd.read_csv(fa[s]), pd.read_csv(fb[s]))
        for name, ps in (("a", psa), ("b", psb)):
            if s in ps.index:
                st["base_idsw"] = int(ps.loc[s, "base_idsw"])
                st[f"idsw_{name}"] = int(ps.loc[s, "smp_idsw"])
                if "switches_fixed" in ps:
                    st[f"fixed_{name}"] = int(ps.loc[s, "switches_fixed"])
                    st[f"introduced_{name}"] = int(ps.loc[s, "switches_introduced"])
        rows.append(dict(seq=s, **st))
        confs.append(conf)
    df = pd.DataFrame(rows).set_index("seq")
    tot = df.sum(numeric_only=True)
    conf = pd.concat(confs).groupby(level=0).sum().fillna(0).astype(int) if confs else pd.DataFrame()
    a, b = args.a_name, args.b_name
    lines = [f"# {a} vs {b}: window-by-window agreement", "",
             f"Sequences: {', '.join(seqs)}", "",
             f"- Matched windows: {int(tot.matched)} (only in {a}: {int(tot.only_a)}, only in {b}: {int(tot.only_b)})",
             f"- Windows with a seed in both runs: {int(tot.seeded)}; VOS ran in both: {int(tot.ran_both)}",
             f"- Ran in {b} but not in {a}: {int(tot.ran_b_not_a)} (the oracle returns nothing when the seed box "
             f"matches no GT object, e.g. a YOLO box without annotation)",
             f"- Same outcome when both ran: {int(tot.same_outcome_ran_both)}/{int(tot.ran_both)}",
             f"- Same SWAP / no-SWAP decision when both ran: {int(tot.same_swap_decision_ran_both)}/{int(tot.ran_both)}",
             f"- SWAP outcomes: {a} {int(tot.swaps_a)}, {b} {int(tot.swaps_b)}; VOS frames: {a} {int(tot.vos_frames_a)}, "
             f"{b} {int(tot.vos_frames_b)}"]
    if "idsw_a" in df and "idsw_b" in df:
        lines.append(f"- IDSW: ByteTrack {int(tot.base_idsw)} -> {a} {int(tot.idsw_a)}, {b} {int(tot.idsw_b)}")
    lines += ["", "Per sequence:", "", df.to_markdown(), "", f"Outcome confusion (rows {a}, columns {b}, seeded windows):",
              "", conf.to_markdown() if len(conf) else "(none)"]
    report = "\n".join(lines)
    print(report)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        df.to_csv(os.path.join(args.out, "per_sequence.csv"))
        conf.to_csv(os.path.join(args.out, "outcome_confusion.csv"))
        with open(os.path.join(args.out, "report.md"), "w") as fh:
            fh.write(report + "\n")


if __name__ == "__main__":
    main()
