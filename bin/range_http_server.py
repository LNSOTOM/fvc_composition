#!/usr/bin/env python3
"""Static file server with HTTP Range support.

Why this exists:
- `python -m http.server` (at least on Python 3.10.x) serves full files even when
  the client requests `Range: bytes=...`.
- COG readers (geotiff.js/georaster) rely on HTTP Range (206 Partial Content)
  to stream GeoTIFF tiles efficiently.

Usage:
  python3 bin/range_http_server.py 8001
# then open http://127.0.0.1:8001/cnn_mappingAI_viewer.html

Notes:
- Supports single-range requests (the common case for COG).
- Falls back to default behavior when no Range header is provided.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import IO, Optional, Tuple


_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


class RangeRequestHandler(SimpleHTTPRequestHandler):
    # Silence overly chatty logging; keep errors.
    def log_message(self, format: str, *args) -> None:  # noqa: A003 (shadow builtin)
        super().log_message(format, *args)

    def send_head(self) -> Optional[IO[bytes]]:
        path = self.translate_path(self.path)

        # georaster/geotiff.js probes for external overview sidecars via `<tif>.ovr`.
        # Our COGs already contain internal overviews, so the sidecar is optional.
        # Returning 204 here prevents noisy DevTools 404 errors without changing behavior.
        if path.endswith(".ovr") and not os.path.exists(path):
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None

        if os.path.isdir(path):
            return super().send_head()

        range_header = self.headers.get("Range")
        if not range_header:
            return super().send_head()

        match = _RANGE_RE.match(range_header.strip())
        if not match:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Invalid Range header")
            return None

        try:
            file_handle = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None

        try:
            file_size = os.fstat(file_handle.fileno()).st_size
            start_s, end_s = match.groups()

            if start_s == "" and end_s == "":
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Invalid Range")
                file_handle.close()
                return None

            if start_s == "":
                # Suffix range: last N bytes
                suffix_len = int(end_s)
                if suffix_len <= 0:
                    self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Invalid Range")
                    file_handle.close()
                    return None
                start = max(0, file_size - suffix_len)
                end = file_size - 1
            else:
                start = int(start_s)
                end = int(end_s) if end_s != "" else file_size - 1

            if start >= file_size or end < start:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Range out of bounds")
                file_handle.close()
                return None

            end = min(end, file_size - 1)
            length = end - start + 1

            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-type", self.guess_type(path))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Last-Modified", self.date_time_string(os.path.getmtime(path)))
            self.end_headers()

            file_handle.seek(start)
            self._range: Optional[Tuple[int, int]] = (start, end)
            return file_handle
        except Exception:
            file_handle.close()
            raise

    def copyfile(self, source: IO[bytes], outputfile: IO[bytes]) -> None:
        byte_range = getattr(self, "_range", None)
        if not byte_range:
            shutil.copyfileobj(source, outputfile)
            return

        start, end = byte_range
        remaining = end - start + 1
        bufsize = 64 * 1024
        while remaining > 0:
            chunk = source.read(min(bufsize, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Static HTTP server with Range support")
    parser.add_argument("port", type=int, nargs="?", default=8001)
    parser.add_argument("--bind", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.bind, args.port), RangeRequestHandler)
    print(f"Serving with Range support at http://{args.bind}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
