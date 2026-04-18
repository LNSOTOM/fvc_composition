#!/usr/bin/env python3
from __future__ import annotations

import mimetypes
import os
from pathlib import Path, PurePosixPath

from flask import Flask, Response, abort, redirect, send_file


REPO_ROOT = Path(__file__).resolve().parent
VIEWER_ROUTE = "/fvc_composition-viewer"
VIEWER_INDEX = "cnn_mappingAI_viewer.html"
DEFAULT_PORT = 8001
DEFAULT_HOST = "0.0.0.0"
DEFAULT_CACHE_SECONDS = 0


mimetypes.add_type("application/geo+json", ".geojson")
mimetypes.add_type("image/tiff", ".tif")
mimetypes.add_type("image/tiff", ".tiff")


app = Flask(__name__)


def get_cache_seconds() -> int:
    raw_value = os.environ.get("FVC_VIEWER_CACHE_SECONDS", str(DEFAULT_CACHE_SECONDS))
    try:
        return max(0, int(raw_value))
    except (TypeError, ValueError):
        return DEFAULT_CACHE_SECONDS


def normalize_relative_path(relative_path: str) -> str:
    normalized = PurePosixPath("/" + str(relative_path or "")).relative_to("/")
    if any(part == ".." for part in normalized.parts):
        abort(404)
    cleaned = normalized.as_posix()
    if cleaned in {"", "."}:
        abort(404)
    return cleaned


def resolve_repo_path(relative_path: str) -> Path:
    cleaned = normalize_relative_path(relative_path)
    candidate = REPO_ROOT / cleaned
    if candidate.is_dir():
        abort(404)
    return candidate


def apply_cache_headers(response: Response) -> Response:
    cache_seconds = get_cache_seconds()
    if cache_seconds > 0:
        response.headers["Cache-Control"] = f"public, max-age={cache_seconds}"
    else:
        response.headers["Cache-Control"] = "no-cache"
    response.headers.setdefault("Accept-Ranges", "bytes")
    return response


def serve_repo_file(relative_path: str) -> Response:
    file_path = resolve_repo_path(relative_path)

    if not file_path.exists():
        if str(file_path).endswith(".ovr"):
            return apply_cache_headers(Response(status=204))
        abort(404)

    response = send_file(file_path, conditional=True, etag=True, max_age=get_cache_seconds())
    return apply_cache_headers(response)


@app.get("/")
def root() -> Response:
    return redirect(f"{VIEWER_ROUTE}/", code=302)


@app.get("/_health")
def healthcheck() -> tuple[str, int]:
    return "ok", 200


@app.get(VIEWER_ROUTE)
def viewer_root() -> Response:
    return redirect(f"{VIEWER_ROUTE}/", code=308)


@app.route(f"{VIEWER_ROUTE}/", methods=["GET", "HEAD"])
def viewer_index() -> Response:
    return serve_repo_file(VIEWER_INDEX)


@app.route(f"{VIEWER_ROUTE}/<path:relative_path>", methods=["GET", "HEAD"])
def viewer_assets(relative_path: str) -> Response:
    return serve_repo_file(relative_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", str(DEFAULT_PORT)))
    host = os.environ.get("FVC_VIEWER_HOST", DEFAULT_HOST)
    app.run(host=host, port=port, debug=False)