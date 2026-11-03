#!/usr/bin/env python
"""Download the FastTracker benchmark from Hugging Face into <out>/train/<sequence>/.

The dataset ships one .zip per sequence (0.5-1.6 GB each) containing
img1/*.jpg, gt/gt.txt, video/*.mp4 and seqinfo.ini.

  --mode gt    (default, CPU stage) only gt/gt.txt + seqinfo.ini of every sequence.
               Reads them straight out of the remote zips with HTTP Range requests
               (a few MB in total). If the server refuses ranges, it falls back to
               downloading one zip at a time, extracting the two files and deleting
               the zip.
  --mode full  (GPU stage) img1 frames + gt + seqinfo (video/ skipped unless
               --keep-video). One zip at a time, deleted after extraction, so the
               peak extra disk use is one archive.

Examples:
    python scripts/download_fasttracker.py --out /content/datasets/FastTracker
    python scripts/download_fasttracker.py --out /content/datasets/FastTracker --mode full --seqs task_tunnel
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rad.io.remote_zip import HttpRangeFile, RangeNotSupported  # noqa: E402

REPO = "Hamidreza-Hashemp/FastTracker-Benchmark"
MARKERS = ("gt/", "img1/", "video/", "det/")


def member_target(name: str) -> str | None:
    """Path of a zip member relative to the sequence folder (None = skip).

    Works whether or not the archive has a top-level folder:
    'task_x/gt/gt.txt' -> 'gt/gt.txt', 'gt/gt.txt' -> 'gt/gt.txt', 'a/b/seqinfo.ini' -> 'seqinfo.ini'.
    """
    n = name.replace("\\", "/")
    if n.endswith("/") or "__MACOSX" in n or os.path.basename(n).startswith("._"):
        return None
    if os.path.basename(n) == "seqinfo.ini":
        return "seqinfo.ini"
    for m in MARKERS:
        k = n.find(m)
        if k == 0 or (k > 0 and n[k - 1] == "/"):
            return n[k:]
    return None


def wanted(rel: str, mode: str, keep_video: bool) -> bool:
    if rel in ("seqinfo.ini", "gt/gt.txt"):
        return True
    if mode == "full":
        if rel.startswith("img1/") or rel.startswith("det/"):
            return True
        if rel.startswith("video/") and keep_video:
            return True
    return False


def extract(zf: zipfile.ZipFile, seq_dir: str, mode: str, keep_video: bool) -> int:
    n = 0
    for info in zf.infolist():
        rel = member_target(info.filename)
        if rel is None or not wanted(rel, mode, keep_video):
            continue
        dst = os.path.join(seq_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with zf.open(info) as src, open(dst, "wb") as out:
            shutil.copyfileobj(src, out, 1 << 20)
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="dataset root; sequences go to <out>/train/<name>")
    ap.add_argument("--mode", choices=["gt", "full"], default="gt")
    ap.add_argument("--seqs", nargs="*", default=None, help="subset of sequence names (zip stems)")
    ap.add_argument("--keep-video", action="store_true")
    ap.add_argument("--no-range", action="store_true", help="always download whole zips")
    ap.add_argument("--tmp", default=None, help="where to put a zip while extracting (default: next to --out)")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
    try:
        from huggingface_hub import get_token
    except ImportError:  # older huggingface_hub
        def get_token():
            return os.environ.get("HF_TOKEN")

    files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.endswith(".zip")]
    if args.seqs:
        keep = set(args.seqs)
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in keep]
    if not files:
        sys.exit("no matching .zip files in the dataset")
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    tmp_root = args.tmp or os.path.join(args.out, "_tmp")
    os.makedirs(tmp_root, exist_ok=True)
    use_range = args.mode == "gt" and not args.no_range
    for f in files:
        seq = os.path.splitext(os.path.basename(f))[0]
        split = os.path.basename(os.path.dirname(f)) or "train"
        seq_dir = os.path.join(args.out, split, seq)
        os.makedirs(seq_dir, exist_ok=True)
        t = time.time()
        if use_range:
            try:
                rf = HttpRangeFile(hf_hub_url(REPO, f, repo_type="dataset"), headers=headers)
                with zipfile.ZipFile(rf) as zf:
                    n = extract(zf, seq_dir, args.mode, args.keep_video)
                print(f"{seq}: {n} files via HTTP range ({rf.bytes_fetched / 1e6:.1f} MB read, {time.time() - t:.0f}s)")
                continue
            except (RangeNotSupported, OSError, zipfile.BadZipFile) as e:
                print(f"{seq}: range read failed ({e}); downloading the whole zip instead")
                use_range = False
        d = tempfile.mkdtemp(dir=tmp_root)
        try:
            path = hf_hub_download(REPO, f, repo_type="dataset", local_dir=d)
            with zipfile.ZipFile(path) as zf:
                n = extract(zf, seq_dir, args.mode, args.keep_video)
            print(f"{seq}: {n} files from the full zip ({os.path.getsize(path) / 1e9:.2f} GB, {time.time() - t:.0f}s)")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(tmp_root, ignore_errors=True)
    print("done:", os.path.join(args.out, "train"))


if __name__ == "__main__":
    main()
