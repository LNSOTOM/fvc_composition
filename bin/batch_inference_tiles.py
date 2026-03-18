#!/usr/bin/env python3
"""Batch inference workflow for many tiles.

For each input tile GeoTIFF (e.g. tiles_multispectral.22.tif), this script:

- stages the input into a per-tile output folder (copy/symlink/hardlink)
- reprojects predictor to EPSG:4326 (GeoTIFF)
- converts predictor to a Cloud-Optimized GeoTIFF (COG)
- runs inference and exports polygons to predictions.geojson
- generates a small PNG thumbnail for the viewer
- generates STAC metadata (stac/ folder)

It also writes tiles_index.json (repo-root by default) so ai_assist_viewer.html can
populate the tile selector dynamically.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import errno
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TILE_RE = re.compile(r"tiles_multispectral\.(\d+)\.tif$", re.IGNORECASE)


@dataclass(frozen=True)
class TilePaths:
    tile_id: str
    out_dir: Path
    predictor_raw: Path
    predictor_epsg4326: Path
    predictor_cog: Path
    predictions_geojson: Path
    thumbnail_png: Path
    stac_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference workflow for many tiles")

    parser.add_argument(
        "--python",
        default="",
        help=(
            "Python interpreter to use for running repo scripts (inference/thumbnail/STAC). "
            "Defaults to the interpreter running this batch script."
        ),
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing tiles_multispectral.<id>.tif files",
    )
    parser.add_argument(
        "--tile-ids",
        default="",
        help="Optional comma-separated tile ids to run (e.g. 22,55). If omitted, runs all found.",
    )

    parser.add_argument(
        "--output-root",
        default="phase_3_models/unet_site_models",
        help="Root folder for outputs (default: phase_3_models/unet_site_models)",
    )
    parser.add_argument(
        "--output-prefix",
        default="wombat_predictions_stitch_medium_1024_120ep_raw_bestmodel_55_tile",
        help="Output folder name prefix under output-root (default matches existing tile22 folder)",
    )

    parser.add_argument(
        "--model-path",
        default="phase_3_models/unet_site_models/outputs_ecosystems/medium/original/block_2_epoch_55.pth",
        help="Model checkpoint to use",
    )
    parser.add_argument("--in-channels", type=int, default=5, help="Model input channels (default: 5)")
    parser.add_argument("--valid-classes", default="0,1,2,3", help="Valid class ids (default: 0,1,2,3)")

    parser.add_argument(
        "--stage-mode",
        choices=["copy", "symlink", "hardlink"],
        default="symlink",
        help="How to place predictor_tile_<id>.tif into output dir (default: symlink)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs (otherwise steps are skipped if outputs exist)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing other tiles if one tile fails",
    )

    parser.add_argument(
        "--write-tiles-index",
        default="tiles_index.json",
        help="Write a JSON index of tiles (default: tiles_index.json in repo root). Use empty to disable.",
    )

    return parser.parse_args()


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _stage_input(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    _ensure_parent(dst)

    if dst.exists() or dst.is_symlink():
        if overwrite:
            try:
                dst.unlink()
            except IsADirectoryError:
                shutil.rmtree(dst)
        else:
            return

    def do_copy() -> None:
        shutil.copy2(src, dst)

    if mode == "copy":
        do_copy()
        return

    try:
        if mode == "symlink":
            dst.symlink_to(src)
            return
        if mode == "hardlink":
            os.link(src, dst)
            return
        raise ValueError(f"Unknown stage mode: {mode}")
    except OSError as exc:
        # Common on exFAT/NTFS mounts or restricted environments.
        if exc.errno in {
            errno.EPERM,
            errno.EOPNOTSUPP,
            getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
        }:
            print(
                f"WARN: {mode} not permitted for {dst} (errno={exc.errno}). Falling back to copy.",
                file=sys.stderr,
            )
            do_copy()
            return
        raise


def discover_tiles(input_dir: Path) -> dict[str, Path]:
    tiles: dict[str, Path] = {}
    for p in sorted(input_dir.glob("tiles_multispectral.*.tif")):
        m = TILE_RE.search(p.name)
        if not m:
            continue
        tile_id = m.group(1)
        tiles[tile_id] = p
    return tiles


def parse_tile_id_filter(raw: str) -> set[str]:
    raw = (raw or "").strip()
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def build_tile_paths(tile_id: str, output_root: Path, output_prefix: str) -> TilePaths:
    out_dir = output_root / f"{output_prefix}{tile_id}"
    return TilePaths(
        tile_id=tile_id,
        out_dir=out_dir,
        predictor_raw=out_dir / f"predictor_tile_{tile_id}.tif",
        predictor_epsg4326=out_dir / f"predictor_tile_{tile_id}_epsg4326.tif",
        predictor_cog=out_dir / f"predictor_tile_{tile_id}_epsg4326_cog.tif",
        predictions_geojson=out_dir / "predictions.geojson",
        thumbnail_png=out_dir / "thumbnail_531.png",
        stac_dir=out_dir / "stac",
    )


def step_reproject_to_epsg4326(paths: TilePaths, cwd: Path, overwrite: bool) -> None:
    if paths.predictor_epsg4326.exists() and not overwrite:
        return

    _ensure_parent(paths.predictor_epsg4326)
    _run(
        [
            "gdalwarp",
            "-overwrite" if overwrite else "-overwrite",
            "-t_srs",
            "EPSG:4326",
            "-r",
            "bilinear",
            "-multi",
            "-wo",
            "NUM_THREADS=ALL_CPUS",
            str(paths.predictor_raw),
            str(paths.predictor_epsg4326),
        ],
        cwd=cwd,
    )


def step_build_cog(paths: TilePaths, cwd: Path, overwrite: bool) -> None:
    if paths.predictor_cog.exists() and not overwrite:
        return

    _ensure_parent(paths.predictor_cog)
    _run(
        [
            "gdal_translate",
            "-of",
            "COG",
            "-co",
            "COMPRESS=DEFLATE",
            "-co",
            "LEVEL=9",
            "-co",
            "PREDICTOR=FLOATING_POINT",
            "-co",
            "OVERVIEW_COMPRESS=DEFLATE",
            "-co",
            "OVERVIEW_PREDICTOR=FLOATING_POINT",
            "-co",
            "RESAMPLING=BILINEAR",
            "-co",
            "OVERVIEWS=AUTO",
            "-co",
            "NUM_THREADS=ALL_CPUS",
            "-co",
            "BIGTIFF=IF_SAFER",
            str(paths.predictor_epsg4326),
            str(paths.predictor_cog),
        ],
        cwd=cwd,
    )


def step_inference(
    paths: TilePaths,
    cwd: Path,
    python_exe: str,
    model_path: Path,
    in_channels: int,
    valid_classes: str,
    overwrite: bool,
) -> None:
    if paths.predictions_geojson.exists() and not overwrite:
        return

    _ensure_parent(paths.predictions_geojson)
    script = cwd / "phase_3_models" / "unet_site_models" / "inference_raster_to_geojson.py"

    cmd = [
        python_exe,
        str(script),
        "--model-path",
        str(model_path),
        "--input-raster",
        str(paths.predictor_raw),
        "--output-geojson",
        str(paths.predictions_geojson),
        "--in-channels",
        str(in_channels),
        "--valid-classes",
        valid_classes,
    ]

    _run(cmd, cwd=cwd)


def step_thumbnail(paths: TilePaths, cwd: Path, python_exe: str, overwrite: bool) -> None:
    if paths.thumbnail_png.exists() and not overwrite:
        return

    script = cwd / "bin" / "generate_tile_thumbnail.py"
    cmd = [
        python_exe,
        str(script),
        "--input",
        str(paths.predictor_cog),
        "--output",
        str(paths.thumbnail_png),
        "--size",
        "240",
    ]
    _run(cmd, cwd=cwd)


def step_stac(paths: TilePaths, cwd: Path, python_exe: str, overwrite: bool) -> None:
    item = paths.stac_dir / f"item_tile{paths.tile_id}.json"
    if item.exists() and not overwrite:
        return

    script = cwd / "bin" / "generate_tile_stac.py"
    cmd = [
        python_exe,
        str(script),
        "--tile-id",
        paths.tile_id,
        "--tile-dir",
        str(paths.out_dir),
    ]
    _run(cmd, cwd=cwd)


def write_tiles_index(index_path: Path, tile_ids: Iterable[str], output_root: Path, output_prefix: str) -> None:
    tiles = []
    for tile_id in sorted(tile_ids, key=lambda x: int(x) if x.isdigit() else x):
        out_dir = output_root / f"{output_prefix}{tile_id}"
        tiles.append(
            {
                "tile_id": str(tile_id),
                "base_path": str(out_dir.as_posix()) + "/",
                "thumbnail": str((out_dir / "thumbnail_531.png").as_posix()),
            }
        )

    payload = {
        "schema_version": 1,
        "tiles": tiles,
    }
    index_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    cwd = Path.cwd()

    python_exe = str(Path(args.python)) if args.python else sys.executable

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    model_path = Path(args.model_path)

    if not input_dir.exists():
        raise FileNotFoundError(f"--input-dir not found: {input_dir}")

    tiles = discover_tiles(input_dir)
    if not tiles:
        raise RuntimeError(f"No tiles found in {input_dir} matching tiles_multispectral.<id>.tif")

    allowed = parse_tile_id_filter(args.tile_ids)
    tile_ids = [tid for tid in tiles.keys() if (not allowed or tid in allowed)]

    if not tile_ids:
        raise RuntimeError("No tiles to process after applying --tile-ids filter")

    # Validate that the chosen interpreter can import torch before starting inference.
    # (Many systems have /usr/bin/python without torch even when a conda env exists.)
    try:
        subprocess.run(
            [python_exe, "-c", "import torch; print(torch.__version__)"],
            cwd=str(cwd),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        print(
            "ERROR: Selected Python interpreter cannot import 'torch'.\n"
            f"  interpreter: {python_exe}\n"
            "Run this script using your conda env Python (or pass --python). Examples:\n"
            "  conda run -n <env> python bin/batch_inference_tiles.py ...\n"
            "  python bin/batch_inference_tiles.py --python $(which python) ...",
            file=sys.stderr,
        )
        return 2

    processed: list[str] = []

    for tile_id in sorted(tile_ids, key=lambda x: int(x) if x.isdigit() else x):
        src = tiles[tile_id]
        paths = build_tile_paths(tile_id, output_root=output_root, output_prefix=args.output_prefix)

        try:
            paths.out_dir.mkdir(parents=True, exist_ok=True)

            _stage_input(src, paths.predictor_raw, mode=args.stage_mode, overwrite=args.overwrite)
            step_reproject_to_epsg4326(paths, cwd=cwd, overwrite=args.overwrite)
            step_build_cog(paths, cwd=cwd, overwrite=args.overwrite)
            step_inference(
                paths,
                cwd=cwd,
                python_exe=python_exe,
                model_path=model_path,
                in_channels=int(args.in_channels),
                valid_classes=str(args.valid_classes),
                overwrite=args.overwrite,
            )
            step_thumbnail(paths, cwd=cwd, python_exe=python_exe, overwrite=args.overwrite)
            step_stac(paths, cwd=cwd, python_exe=python_exe, overwrite=args.overwrite)

            processed.append(tile_id)
            print(f"OK tile {tile_id}: {paths.out_dir}")

        except Exception as exc:
            print(f"FAIL tile {tile_id}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                raise

    if args.write_tiles_index:
        index_path = Path(args.write_tiles_index)
        write_tiles_index(index_path, processed, output_root=output_root, output_prefix=args.output_prefix)
        print(f"Wrote tiles index: {index_path}")

    print(f"Done. Processed {len(processed)} tiles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
