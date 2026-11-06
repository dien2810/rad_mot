#!/usr/bin/env python
"""Compare two policy sweeps run on the same detections, e.g. oracle VOS vs SAM 3.

Checks whether the cost/accuracy result measured with the cheap oracle VOS holds with
SAM 3: for every sequence and policy present in both sweeps, VOS frames, windows, IDSW
and IDF1 side by side, then totals and the RAD/SMP frame ratio under each backend.

Example:
    python scripts/compare_sweeps.py --a outputs/dispatch/sweep_per_sequence.csv \
        --b outputs/dispatch_sam3/sweep_per_sequence.csv --out outputs/compare_sweeps
A file name may also be a folder (sweep_per_sequence.csv or, if the run died,
sweep_per_sequence.partial.csv inside it); several --b files are concatenated, so runs
made one sequence at a time can be combined.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

KEY = ["seq", "policy", "param"]


def read_sweep(paths: list[str]) -> pd.DataFrame:
    parts = []
    for p in paths:
        if os.path.isdir(p):
            for name in ("sweep_per_sequence.csv", "sweep_per_sequence.partial.csv"):
                if os.path.exists(os.path.join(p, name)):
                    p = os.path.join(p, name)
                    break
            else:
                raise FileNotFoundError(f"no sweep_per_sequence(.partial).csv in {p}")
        parts.append(pd.read_csv(p))
    df = pd.concat(parts, ignore_index=True)
    if "seed" in df:  # random policy: one row per seed -> mean
        df = df.groupby(KEY, as_index=False).mean(numeric_only=True)
    df["param"] = df["param"].round(4)
    return df.drop_duplicates(KEY, keep="last")


def compare(a: pd.DataFrame, b: pd.DataFrame, a_name: str = "oracle", b_name: str = "sam3") -> dict:
    cols = ["vos_frames", "windows", "idsw", "idf1"]
    m = a[KEY + cols].merge(b[KEY + cols], on=KEY, suffixes=(f"_{a_name}", f"_{b_name}"))
    if m.empty:
        raise SystemExit("no (sequence, policy, param) in common")
    tot = m.groupby(["policy", "param"]).agg(
        seqs=("seq", "nunique"), **{f"{c}_{n}": (f"{c}_{n}", "sum") for c in cols[:3] for n in (a_name, b_name)})
    out = dict(rows=m.sort_values(KEY).reset_index(drop=True), totals=tot.reset_index())
    t = tot.reset_index().set_index(["policy", "param"])
    if ("smp", 0.05) in t.index and ("learned", 1.0) in t.index:
        smp, rad = t.loc[("smp", 0.05)], t.loc[("learned", 1.0)]
        out["ratio"] = {n: float(rad[f"vos_frames_{n}"] / max(smp[f"vos_frames_{n}"], 1)) for n in (a_name, b_name)}
        out["idsw"] = {n: (int(smp[f"idsw_{n}"]), int(rad[f"idsw_{n}"])) for n in (a_name, b_name)}
    return out


def report(res: dict, a_name: str, b_name: str) -> str:
    lines = [f"# {a_name} vs {b_name}: policy sweep on the same detections", "",
             f"Sequences: {', '.join(sorted(res['rows'].seq.unique()))}", ""]
    if "ratio" in res:
        for n in (a_name, b_name):
            s, r = res["idsw"][n]
            lines.append(f"- {n}: RAD/SMP VOS frames = {res['ratio'][n]:.2f}; IDSW SMP {s}, RAD {r}")
        lines.append("")
    lines += ["Totals:", "", res["totals"].to_markdown(index=False), "", "Per sequence:", "",
              res["rows"].round(4).to_markdown(index=False)]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", nargs="+", required=True, help="sweep CSV(s) or folder(s) of backend A (oracle)")
    ap.add_argument("--b", nargs="+", required=True, help="sweep CSV(s) or folder(s) of backend B (SAM 3)")
    ap.add_argument("--a-name", default="oracle")
    ap.add_argument("--b-name", default="sam3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = compare(read_sweep(args.a), read_sweep(args.b), args.a_name, args.b_name)
    text = report(res, args.a_name, args.b_name)
    print(text)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        res["rows"].to_csv(os.path.join(args.out, "per_sequence.csv"), index=False)
        res["totals"].to_csv(os.path.join(args.out, "totals.csv"), index=False)
        with open(os.path.join(args.out, "report.md"), "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
