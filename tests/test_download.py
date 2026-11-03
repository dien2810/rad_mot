"""Remote-zip reading over HTTP Range and member path mapping (local test server)."""
import http.server
import io
import os
import sys
import threading
import zipfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from rad.io.remote_zip import HttpRangeFile, RangeNotSupported  # noqa: E402
from download_fasttracker import extract, member_target  # noqa: E402


def _make_zip(prefix: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr(prefix + "seqinfo.ini", "[Sequence]\nseqLength=3\n")
        zf.writestr(prefix + "gt/gt.txt", "1,1,10,10,5,5,1,3,1.0\n")
        for i in range(3):
            zf.writestr(prefix + f"img1/{i + 1:06d}.jpg", os.urandom(300_000))
        zf.writestr(prefix + "video/v.mp4", os.urandom(500_000))
    return buf.getvalue()


def _server(payload: bytes, ranges: bool):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            if ranges and rng:
                a, b = rng.split("=")[1].split("-")
                a, b = int(a), min(int(b), len(payload) - 1)
                body = payload[a:b + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {a}-{b}/{len(payload)}")
            else:
                body = payload
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/x.zip"


def test_member_target():
    assert member_target("task_a/gt/gt.txt") == "gt/gt.txt"
    assert member_target("gt/gt.txt") == "gt/gt.txt"
    assert member_target("a/b/seqinfo.ini") == "seqinfo.ini"
    assert member_target("task_a/img1/000001.jpg") == "img1/000001.jpg"
    assert member_target("__MACOSX/task_a/gt/._gt.txt") is None
    assert member_target("task_a/") is None
    assert member_target("task_agt/x") is None


@pytest.mark.parametrize("prefix", ["", "task_a/"])
def test_gt_only_via_range(tmp_path, prefix, monkeypatch):
    for k in ("HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    payload = _make_zip(prefix)
    srv, url = _server(payload, ranges=True)
    try:
        rf = HttpRangeFile(url, block=64 * 1024)
        with zipfile.ZipFile(rf) as zf:
            n = extract(zf, str(tmp_path / "S"), "gt", keep_video=False)
        assert n == 2
        assert (tmp_path / "S" / "gt" / "gt.txt").read_text().startswith("1,1,")
        assert (tmp_path / "S" / "seqinfo.ini").exists()
        assert not (tmp_path / "S" / "img1").exists()
        assert rf.bytes_fetched < len(payload) / 2  # did not download the images
    finally:
        srv.shutdown()


def test_full_mode_skips_video(tmp_path):
    with zipfile.ZipFile(io.BytesIO(_make_zip("task_a/"))) as zf:
        n = extract(zf, str(tmp_path / "S"), "full", keep_video=False)
    assert n == 5 and len(os.listdir(tmp_path / "S" / "img1")) == 3
    assert not (tmp_path / "S" / "video").exists()


def test_no_range_support_raises(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    srv, url = _server(_make_zip(""), ranges=False)
    try:
        with pytest.raises(RangeNotSupported):
            HttpRangeFile(url)
    finally:
        srv.shutdown()
