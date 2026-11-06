#!/usr/bin/env python
"""Demo app (Gradio): ByteTrack vs SMP vs RAD on a FastTracker clip or on a new video.

Run on Colab (GPU runtime for SAM 3):
    !pip -q install gradio
    !python scripts/demo_app.py --data $DATA --work $WORK --share
then open the gradio.live link that is printed.

Tab "Video FastTracker" (has ground truth): pick a sequence that has img1/ frames, a start
    frame and a length (or let make_demo.py pick a clip around a fixed ID switch). ID
    switches are counted against the ground truth.
Tab "Video mới" (no ground truth): upload a video; frames are extracted (at most --max-seconds),
    YOLO11 detects, and the same pipeline runs. Without labels the app shows windows and
    swaps but cannot count ID switches.

Both run scripts/make_demo.py in a subprocess (its log is streamed to the page) and return
demo.mp4 (2x2 video), the summary and events, and bundle.json + clip.mp4 for the browser
page demo/index.html. The risk model of RAD is trained on the FastTracker candidates of an
earlier run_dispatch.py run (--train-cands), so it never sees the uploaded video.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from typing import Iterator, Optional

import cv2
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rad.io.mot import discover_sequences  # noqa: E402

PY = sys.executable
METHODS = (("base", "ByteTrack"), ("smp", "SMP"), ("rad", "RAD"))


# ----------------------------------------------------------------------------- helpers
def sequences_with_frames(data: Optional[str]) -> list[str]:
    if not data or not os.path.isdir(data):
        return []
    return [s.name for s in discover_sequences(data) if s.frame_paths()]


def stream(cmd: list[str], log: list[str]) -> Iterator[str]:
    """Run a command, yield the growing log; raise if it fails."""
    log.append("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"})
    for line in proc.stdout:
        line = line.rstrip()
        if not line or "frame loading" in line:  # SAM 3's per-frame progress bars
            continue
        log.append(line)
        yield "\n".join(log[-400:])
    if proc.wait() != 0:
        raise RuntimeError(f"{os.path.basename(cmd[1])} failed (exit {proc.returncode}); see the log")


def extract_frames(video: str, out_root: str, name: str, start_s: float, seconds: float,
                   max_side: int = 1920) -> tuple[str, int, float]:
    """Write <out_root>/train/<name>/img1/000001.jpg ... and seqinfo.ini. Returns (data dir, frames, fps)."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start_s > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
    seq_dir = os.path.join(out_root, "train", name)
    img_dir = os.path.join(seq_dir, "img1")
    os.makedirs(img_dir, exist_ok=True)
    n, size = 0, None
    limit = int(round(seconds * fps))
    while n < limit:
        ok, img = cap.read()
        if not ok:
            break
        h, w = img.shape[:2]
        s = min(1.0, max_side / max(h, w))
        if s < 1.0:
            img = cv2.resize(img, (int(round(w * s)) // 2 * 2, int(round(h * s)) // 2 * 2), interpolation=cv2.INTER_AREA)
        size = img.shape[1], img.shape[0]
        n += 1
        cv2.imwrite(os.path.join(img_dir, f"{n:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cap.release()
    if n == 0:
        raise RuntimeError("no frame could be read from the video")
    with open(os.path.join(seq_dir, "seqinfo.ini"), "w") as fh:
        fh.write(f"[Sequence]\nname={name}\nimDir=img1\nframeRate={fps:.3f}\nseqLength={n}\n"
                 f"imWidth={size[0]}\nimHeight={size[1]}\nimExt=.jpg\n")
    return os.path.join(out_root, "train"), n, fps


def demo_cmd(data: str, seq: str, out: str, backend: str, dets: str, det_dir: Optional[str],
             train_cands: Optional[str], start: Optional[int], frames: int, precision: str) -> list[str]:
    cmd = [PY, "-u", os.path.join(ROOT, "scripts", "make_demo.py"), "--data", data, "--seq", seq,
           "--dets", dets, "--backend", backend, "--out", out, "--width", "1600"]
    if det_dir:
        cmd += ["--det-dir", det_dir]
    if train_cands and glob.glob(train_cands):
        cmd += ["--train-cands", train_cands]
    if start is not None:
        cmd += ["--start", str(int(start)), "--frames", str(int(frames))]
    if backend == "sam3":
        cmd += ["--precision", precision, "--frames-on-gpu"]
    return cmd


def result_dir(log: list[str]) -> str:
    for line in reversed(log):
        m = re.match(r"\[done\] (.+)$", line)
        if m:
            return m.group(1).strip()
    raise RuntimeError("make_demo.py did not report an output folder")


def summarize(out_dir: str) -> tuple[str, pd.DataFrame, list[str], Optional[str]]:
    """(markdown summary, events table, files to download, demo video)."""
    with open(os.path.join(out_dir, "summary.json"), encoding="utf-8") as fh:
        s = json.load(fh)
    has_gt = s.get("has_gt", True)
    pol = s.get("policies", {})
    ev = s.get("clip_eval") or {}
    head = "| Phương pháp | Kích hoạt | Cửa sổ | Cửa sổ chạy VOS | Khung VOS | Hoán đổi áp dụng |" + \
           (" Lỗi đổi ID | IDF1 |" if has_gt else "")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for key, name in METHODS:
        p = pol.get(key, {})
        row = [name, p.get("triggers", 0), p.get("windows", 0), p.get("windows_ran", 0), p.get("vos_frames", 0),
               p.get("swaps_applied", 0)]
        if has_gt and ev.get(key):
            row += [ev[key]["idsw"], f"{ev[key]['idf1']:.3f}".replace(".", ",")]
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    s0, s1 = s["clip"]
    note = [f"**{s['sequence']}**, khung {s0}–{s1}, VOS: {'SAM 3' if s['backend'] == 'sam3' else 'oracle'}, "
            f"{s.get('vos_frames_computed', 0)} khung VOS tính mới ({s.get('vos_seconds', 0)} s).",
            "", "\n".join(lines), ""]
    if not has_gt:
        note.append("Video không có nhãn GT nên không đếm được lỗi đổi ID; bảng sự kiện liệt kê các cửa sổ.")
    if s.get("risk_model"):
        note.append(f"RAD: {s['risk_model']}.")
    events = pd.read_csv(os.path.join(out_dir, "events.csv")) if os.path.exists(os.path.join(out_dir, "events.csv")) \
        else pd.DataFrame()
    files = [os.path.join(out_dir, f) for f in ("bundle.json", "clip.mp4", "demo.mp4", "summary.json", "events.csv")
             if os.path.exists(os.path.join(out_dir, f))]
    video = os.path.join(out_dir, "demo.mp4")
    return "\n".join(note), events, files, video if os.path.exists(video) else None


# ----------------------------------------------------------------------------- handlers
class App:
    def __init__(self, data: Optional[str], work: str, train_cands: Optional[str], dets_dir: Optional[str],
                 yolo: str, precision: str):
        self.data, self.work, self.train_cands = data, work, train_cands
        self.dets_dir, self.yolo, self.precision = dets_dir, yolo, precision
        self.out = os.path.join(work, "outputs", "demo_app")
        os.makedirs(self.out, exist_ok=True)

    def run_fasttracker(self, seq: str, auto: bool, start: float, frames: float, dets: str, backend: str):
        """Yields (log, summary, events, files, video)."""
        log: list[str] = []
        try:
            if not seq:
                raise RuntimeError("chọn một video FastTracker có ảnh (img1/)")
            det_dir = None
            if dets == "YOLO11 (đã chạy sẵn)":
                det_dir = self.dets_dir
                if not det_dir or not os.path.exists(os.path.join(det_dir, f"{seq}.txt")):
                    raise RuntimeError(f"không có {seq}.txt trong {det_dir}; chạy detect_yolo.py hoặc chọn GT")
            cmd = demo_cmd(self.data, seq, self.out, "sam3" if backend.startswith("SAM") else "oracle",
                           "dir" if det_dir else "oracle", det_dir, self.train_cands, None if auto else start,
                           frames, self.precision)
            for text in stream(cmd, log):
                yield text, "Đang chạy…", None, None, None
            summary, events, files, video = summarize(result_dir(log))
            yield "\n".join(log[-400:]), summary, events, files, video
        except Exception as e:  # shown in the page instead of a stack trace
            log.append(f"LỖI: {e}")
            yield "\n".join(log[-400:]), f"**Lỗi:** {e}", None, None, None

    def run_upload(self, video: Optional[str], start_s: float, seconds: float, backend: str):
        log: list[str] = []
        try:
            if not video:
                raise RuntimeError("tải một video lên trước")
            stem = re.sub(r"[^A-Za-z0-9_]+", "_", os.path.splitext(os.path.basename(video))[0])[:40] or "video"
            name = f"upload_{stem}_{time.strftime('%H%M%S')}"
            root = os.path.join(self.out, "uploads", name)
            log.append(f"Tách khung từ {os.path.basename(video)} (từ giây {start_s:g}, tối đa {seconds:g} giây)…")
            yield "\n".join(log), "Đang tách khung…", None, None, None
            data, n, fps = extract_frames(video, root, name, float(start_s), float(seconds))
            log.append(f"{n} khung, {fps:.2f} fps")
            det_dir = os.path.join(root, "dets")
            det_cmd = [PY, "-u", os.path.join(ROOT, "scripts", "detect_yolo.py"), "--data", data, "--weights", self.yolo,
                       "--out", det_dir, "--imgsz", "1280", "--half"]
            for text in stream(det_cmd, log):
                yield text, "Đang phát hiện bằng YOLO11…", None, None, None
            cmd = demo_cmd(data, name, self.out, "sam3" if backend.startswith("SAM") else "oracle", "dir", det_dir,
                           self.train_cands, 1, n, self.precision)
            for text in stream(cmd, log):
                yield text, "Đang chạy ByteTrack, SMP và RAD…", None, None, None
            summary, events, files, vid = summarize(result_dir(log))
            yield "\n".join(log[-400:]), summary, events, files, vid
        except Exception as e:
            log.append(f"LỖI: {e}")
            yield "\n".join(log[-400:]), f"**Lỗi:** {e}", None, None, None


# ----------------------------------------------------------------------------- UI
CSS = """
.gradio-container { max-width: 1400px !important; }
#hdr .eyebrow { color:#1263c6; font-size:13px; font-weight:700; letter-spacing:.11em; text-transform:uppercase; }
#hdr h1 { margin:6px 0 4px; font-size:30px; letter-spacing:-.03em; }
#hdr p { color:#5d6d82; margin:0; }
#scope { border:1px solid #e6b859; background:#fff8e6; color:#7a5211; border-radius:10px; padding:10px 14px; font-size:14px; }
"""


def build_ui(app: App):
    import gradio as gr

    seqs = sequences_with_frames(app.data)
    with gr.Blocks(title="RAD-MOT demo") as ui:
        gr.HTML("""<div id="hdr"><div class="eyebrow">RAD-MOT • FastTracker • Demo chạy thật</div>
<h1>ByteTrack, SMP và RAD trên video của bạn</h1>
<p>Chọn video FastTracker (có nhãn, đếm được lỗi đổi ID) hoặc tải video mới (chỉ xem sự kiện). Mỗi lần chạy gọi
<code>make_demo.py</code>; kết quả gồm video 2×2, bảng tóm tắt và <code>bundle.json</code> + <code>clip.mp4</code>
cho trang <code>demo/index.html</code>.</p></div>""")
        gr.HTML("""<div id="scope"><b>Thời gian chạy (ước lượng, T4, fp16):</b> nạp SAM 3 khoảng 1–2 phút, rồi khoảng 1,3 giây
cho mỗi khung SAM 3. Đoạn 10 giây thường mất 5–15 phút tùy số cửa sổ. VOS oracle chạy trên CPU trong vài giây nhưng chỉ
dùng được với video có nhãn.</div>""")
        with gr.Tab("Video FastTracker (có nhãn GT)"):
            with gr.Row():
                seq = gr.Dropdown(choices=seqs, value=seqs[0] if seqs else None, label="Video",
                                  info="chỉ các video đã tải ảnh (img1/)")
                dets = gr.Radio(["Từ nhãn GT (oracle)", "YOLO11 (đã chạy sẵn)"], value="Từ nhãn GT (oracle)",
                                label="Phát hiện")
                backend = gr.Radio(["SAM 3 (GPU)", "VOS oracle (CPU, nhanh)"], value="SAM 3 (GPU)", label="Mô hình VOS")
            with gr.Row():
                auto = gr.Checkbox(value=True, label="Tự chọn đoạn quanh một lỗi được sửa")
                start = gr.Number(value=1674, precision=0, label="Khung bắt đầu (khi không tự chọn)")
                frames = gr.Number(value=289, precision=0, label="Số khung")
            go1 = gr.Button("Chạy", variant="primary")
        with gr.Tab("Video mới (không có nhãn)"):
            with gr.Row():
                vid_in = gr.Video(label="Tải video lên (mp4)", sources=["upload"])
                with gr.Column():
                    start_s = gr.Number(value=0, label="Bắt đầu từ giây")
                    seconds = gr.Slider(2, 30, value=10, step=1, label="Độ dài (giây)")
                    backend2 = gr.Radio(["SAM 3 (GPU)"], value="SAM 3 (GPU)", label="Mô hình VOS",
                                        info="không có nhãn nên không dùng được VOS oracle")
                    go2 = gr.Button("Chạy", variant="primary")
        summary = gr.Markdown("Chưa chạy.")
        with gr.Row():
            video_out = gr.Video(label="Video kết quả (ByteTrack | SMP / RAD | biểu đồ)")
            files = gr.File(label="Tải về: bundle.json + clip.mp4 cho demo/index.html", file_count="multiple")
        events = gr.Dataframe(label="Sự kiện trong đoạn (lỗi đổi ID, cửa sổ VOS)", wrap=True)
        log = gr.Textbox(label="Nhật ký", lines=14, max_lines=14, autoscroll=True)
        outs = [log, summary, events, files, video_out]
        go1.click(app.run_fasttracker, [seq, auto, start, frames, dets, backend], outs)
        go2.click(app.run_upload, [vid_in, start_s, seconds, backend2], outs)
    return ui


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="FastTracker train folder (sequences with gt/ and img1/)")
    ap.add_argument("--work", default=".", help="results root; the app writes to <work>/outputs/demo_app")
    ap.add_argument("--train-cands", default=None,
                    help='candidates for the risk model (default: "<work>/outputs/dispatch/*/candidates.csv.gz")')
    ap.add_argument("--dets-dir", default=None, help="YOLO detections of FastTracker (default: <work>/outputs/dets)")
    ap.add_argument("--yolo", default="yolo11l.pt", help="detector weights for uploaded videos")
    ap.add_argument("--precision", choices=["auto", "fp16"], default="fp16")
    ap.add_argument("--share", action="store_true", help="public gradio.live link (needed on Colab)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    train = args.train_cands or os.path.join(args.work, "outputs", "dispatch", "*", "candidates.csv.gz")
    if not glob.glob(train):
        print(f"[warn] no candidates match {train}: RAD will be trained on the other FastTracker sequences "
              "(slow) or fall back to -margin")
    app = App(args.data, args.work, train, args.dets_dir or os.path.join(args.work, "outputs", "dets"), args.yolo,
              args.precision)
    ui = build_ui(app)
    import gradio as gr

    kw = dict(share=args.share, server_port=args.port)
    try:  # Gradio 6 takes theme/css in launch(), older versions in Blocks()
        ui.queue().launch(theme=gr.themes.Soft(primary_hue="blue", font=["Arial", "Helvetica", "sans-serif"]), css=CSS, **kw)
    except TypeError:
        ui.queue().launch(**kw)


if __name__ == "__main__":
    main()
