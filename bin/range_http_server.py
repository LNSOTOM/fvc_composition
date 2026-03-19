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
import email.utils
import os
import re
import shutil
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import IO, Optional, Tuple


_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


class RangeRequestHandler(SimpleHTTPRequestHandler):
    quiet: bool = False
    log_partial_content: bool = False
    cache_seconds: int = 0

    def log_message(self, format: str, *args) -> None:  # noqa: A003 (shadow builtin)
        if self.quiet:
            return
        super().log_message(format, *args)

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        if self.quiet:
            return
        if not self.log_partial_content and str(code) == str(HTTPStatus.PARTIAL_CONTENT.value):
            return
        super().log_request(code, size)

    @staticmethod
    def _file_etag(st: os.stat_result) -> str:
        # Weak ETag based on size + mtime_ns. Fast (no hashing) and good enough for local static files.
        return f'W/"{st.st_size}-{getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))}"'

    def _send_cache_control(self) -> None:
        if self.cache_seconds > 0:
            self.send_header("Cache-Control", f"public, max-age={self.cache_seconds}")
        else:
            self.send_header("Cache-Control", "no-cache")

    def _send_etag(self, etag: str) -> None:
        self.send_header("ETag", etag)

    def _send_last_modified(self, last_modified: float) -> None:
        self.send_header("Last-Modified", self.date_time_string(last_modified))

    def end_headers(self) -> None:
        # Add lightweight caching headers for successful responses where we can determine a backing file.
        # Avoid duplicating Last-Modified because SimpleHTTPRequestHandler already sends it for 200 responses.
        # For 206 / 304 we set all cache validators explicitly in send_head.
        if getattr(self, "_skip_auto_cache_headers", False):
            super().end_headers()
            return

        if getattr(self, "_range", None) is None:
            try:
                path = self.translate_path(self.path)
                if os.path.isfile(path):
                    st = os.stat(path)
                    self._send_etag(self._file_etag(st))
                    self._send_cache_control()
            except Exception:
                pass
        super().end_headers()

    def send_head(self) -> Optional[IO[bytes]]:
        path = self.translate_path(self.path)

        # HEAD is used by COG readers as a lightweight probe. For static files, respond like GET
        # would (sans body). SimpleHTTPRequestHandler already handles this correctly.

        # georaster/geotiff.js probes for external overview sidecars via `<tif>.ovr`.
        # Our COGs already contain internal overviews, so the sidecar is optional.
        # Returning 204 here prevents noisy DevTools 404 errors without changing behavior.
        if path.endswith(".ovr") and not os.path.exists(path):
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Content-Length", "0")
            if self.cache_seconds > 0:
                self.send_header("Cache-Control", f"public, max-age={self.cache_seconds}")
            self.end_headers()
            return None

        if os.path.isdir(path):
            return super().send_head()

        try:
            st = os.stat(path)
        except OSError:
            # Let the base handler generate a directory listing / 404, etc.
            return super().send_head()

        etag = self._file_etag(st)
        last_modified = st.st_mtime

        # Handle conditional requests for non-range GETs.
        # For Range requests, clients typically use If-Range.
        if self.command in {"GET", "HEAD"} and not self.headers.get("Range"):
            inm = self.headers.get("If-None-Match")
            if inm is not None and inm.strip() == etag:
                self.send_response(HTTPStatus.NOT_MODIFIED)
                self._skip_auto_cache_headers = True
                self._send_etag(etag)
                self._send_last_modified(last_modified)
                self._send_cache_control()
                self.end_headers()
                return None

        range_header = self.headers.get("Range")
        if not range_header:
            return super().send_head()

        # If-Range: if it doesn't match, ignore Range and serve whole file.
        if_range = self.headers.get("If-Range")
        if if_range:
            ir = if_range.strip()
            etag_match = ir == etag
            date_match = False
            try:
                parsed = email.utils.parsedate_to_datetime(ir)
                date_match = parsed is not None and parsed.timestamp() >= last_modified
            except Exception:
                date_match = False
            if not (etag_match or date_match):
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

            # Set before end_headers so our end_headers() hook doesn't re-add ETag/Cache-Control.
            self._range: Optional[Tuple[int, int]] = (start, end)

            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-type", self.guess_type(path))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self._send_etag(etag)
            self._send_last_modified(last_modified)
            self._send_cache_control()
            self.end_headers()

            file_handle.seek(start)
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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress access logs (errors still surface as exceptions)",
    )
    parser.add_argument(
        "--log-206",
        action="store_true",
        help="Log HTTP 206 Partial Content requests (disabled by default to reduce noise)",
    )
    parser.add_argument(
        "--cache-seconds",
        type=int,
        default=0,
        help="Cache-Control max-age for static files (default: 0 / no-cache)",
    )
    args = parser.parse_args()

    RangeRequestHandler.quiet = bool(args.quiet)
    RangeRequestHandler.log_partial_content = bool(args.log_206)
    RangeRequestHandler.cache_seconds = max(0, int(args.cache_seconds))

    server = ThreadingHTTPServer((args.bind, args.port), RangeRequestHandler)
    print(f"Serving with Range support at http://{args.bind}:{args.port}/")
    if args.quiet:
        print("Access logs: off (--quiet)")
    elif args.log_206:
        print("Access logs: include 206 (--log-206)")
    else:
        print("Access logs: skipping 206 (default)")
    if args.cache_seconds > 0:
        print(f"Cache-Control: public, max-age={args.cache_seconds}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
