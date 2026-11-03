"""Read members of a remote .zip with HTTP Range requests (no full download).

A zip's table of contents sits at the end of the file, so `zipfile` only needs a
few small reads to list members and extract one of them. This lets us pull the
tiny gt.txt / seqinfo.ini files out of multi-GB sequence archives.
"""
from __future__ import annotations

import io
from typing import Optional


class RangeNotSupported(RuntimeError):
    pass


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests, with a small block cache."""

    def __init__(self, url: str, headers: Optional[dict] = None, block: int = 1 << 20, timeout: float = 60):
        import requests

        self._session = requests.Session()
        self._headers = dict(headers or {})
        self.block = block
        self.timeout = timeout
        # resolve redirects once (HF -> CDN) and check that ranges work
        r = self._session.get(url, headers={**self._headers, "Range": "bytes=0-0"}, timeout=timeout,
                              allow_redirects=True, stream=True)
        if r.status_code != 206:
            r.close()
            raise RangeNotSupported(f"server answered {r.status_code} to a Range request")
        total = r.headers.get("Content-Range", "").split("/")[-1]
        r.close()
        if not total.isdigit():
            raise RangeNotSupported("no Content-Range total size")
        self.url = r.url  # final (redirected) URL; auth headers are not sent across hosts
        self.size = int(total)
        self.pos = 0
        self._cache: dict[int, bytes] = {}
        self.bytes_fetched = 0

    # -- io.RawIOBase API --------------------------------------------------------
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        self.pos = max(0, min(self.pos, self.size))
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        """Bytes [start, end] inclusive."""
        r = self._session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=self.timeout)
        if r.status_code != 206:
            raise RangeNotSupported(f"server answered {r.status_code} to a Range request")
        self.bytes_fetched += len(r.content)
        return r.content

    def _block(self, idx: int) -> bytes:
        if idx not in self._cache:
            start = idx * self.block
            end = min(start + self.block, self.size) - 1
            self._cache[idx] = self._fetch(start, end)
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[idx]

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            idx, off = divmod(self.pos, self.block)
            chunk = self._block(idx)[off:off + n]
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)
