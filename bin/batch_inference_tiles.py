#!/usr/bin/env python3
"""Batch inference workflow for many tiles.

For each input tile GeoTIFF (e.g. tiles_multispectral.22.tif), this script:

- stages the input into a per-tile output folder (copy/symlink/hardlink)
- reprojects predictor to EPSG:4326 (GeoTIFF)
- converts predictor to a Cloud-Optimized GeoTIFF (COG)
- runs inference and exports polygons to predictions.geojson
- generates a small PNG thumbnail for the viewer
- generates STAC metadata (stac/ folder)

It also writes tiles_index.json (repo-root by default) so cnn_mappingAI_viewer.html can
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


TILE_RE = re.compile(r"tiles_multispectral[._](\d+)\.tif$", re.IGNORECASE)


@dataclass(frozen=True)
class TilePaths:
    tile_id: str
    out_dir: Path
    predictor_raw: Path
    predictor_epsg4326: Path
    predictor_cog: Path
    predictions_geojson: Path
    predictions_mask_tif: Path
    predictions_shp: Path
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
        "--variant",
        choices=["low", "low_sparse", "medium", "medium_sparse", "dense"],
        default="medium",
        help="Dataset variant label (used for inference defaults + STAC metadata)",
    )

    parser.add_argument(
        "--model-path",
        default="phase_3_models/unet_site_models/outputs_ecosystems/medium/original/block_2_epoch_55.pth",
        help="Model checkpoint to use",
    )
    parser.add_argument("--in-channels", type=int, default=5, help="Model input channels (default: 5)")
    parser.add_argument(
        "--valid-classes",
        default="",
        help=(
            "Valid class ids as a comma-separated list. If omitted, defaults based on --variant "
            "(low: 0,1,2; medium: 0,1,2,3; dense: 0,1,2,3,4)."
        ),
    )

    parser.add_argument(
        "--class-id-map",
        default="",
        help=(
            "Optional class id remapping applied after prediction, before polygonization. "
            "Format: 'src:dst,src2:dst2' (e.g. '1:2,2:1' to swap NPV/PV)."
        ),
    )

    parser.add_argument(
        "--write-mask-tif",
        action="store_true",
        help="Also write a predicted class-mask GeoTIFF (predictions_mask.tif) per tile",
    )

    parser.add_argument(
        "--write-shp",
        action="store_true",
        help="Also write prediction polygons as an ESRI Shapefile (.shp) per tile",
    )

    parser.add_argument(
        "--stage-mode",
        choices=["copy", "symlink", "hardlink", "none"],
        default="symlink",
        help=(
            "How to stage the input into output dir (default: symlink). "
            "Use 'none' to avoid staging/copying the raw tile (saves disk space)."
        ),
    )

    parser.add_argument(
        "--cleanup-intermediates",
        action="store_true",
        help=(
            "Delete large intermediate files after they are no longer needed (epsg4326 tif, "
            "and staged raw tif when applicable). Useful on full disks."
        ),
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

    parser.add_argument(
        "--stac-collection-id",
        default="",
        help=(
            "Optional STAC collection id override. If empty, defaults to wombat-fvc-<variant>. "
            "Example: wombat-fvc-low"
        ),
    )

    return parser.parse_args()


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _stage_input(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    if mode == "none":
        return

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
    patterns = ["tiles_multispectral.*.tif", "tiles_multispectral_*.tif"]
    for pattern in patterns:
        for p in sorted(input_dir.glob(pattern)):
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
        predictions_geojson=out_dir / f"predictions_tile_{tile_id}.geojson",
        predictions_mask_tif=out_dir / f"predictions_mask_tile_{tile_id}.tif",
        predictions_shp=out_dir / f"predictions_tile_{tile_id}.shp",
        thumbnail_png=out_dir / f"thumbnail_531_tile_{tile_id}.png",
        stac_dir=out_dir / "stac",
    )


def step_reproject_to_epsg4326(paths: TilePaths, cwd: Path, overwrite: bool) -> None:
    if (paths.predictor_epsg4326.exists() or paths.predictor_cog.exists()) and not overwrite:
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
    input_raster: Path,
    in_channels: int,
    valid_classes: str,
    variant: str,
    class_id_map: str,
    write_mask_tif: bool,
    write_shp: bool,
    overwrite: bool,
) -> None:
    if not overwrite:
        if paths.predictions_geojson.exists() and (not write_mask_tif or paths.predictions_mask_tif.exists()):
            return

    _ensure_parent(paths.predictions_geojson)
    script = cwd / "phase_3_models" / "unet_site_models" / "inference_raster_to_geojson.py"

    cmd = [
        python_exe,
        str(script),
        "--model-path",
        str(model_path),
        "--input-raster",
        str(input_raster),
        "--output-geojson",
        str(paths.predictions_geojson),
        "--tile-id",
        str(paths.tile_id),
        "--variant",
        str(variant),
        "--in-channels",
        str(in_channels),
        "--valid-classes",
        valid_classes,
    ]

    if write_mask_tif:
        cmd += ["--output-mask", str(paths.predictions_mask_tif)]

    if write_shp:
        cmd += ["--output-shp", str(paths.predictions_shp)]

    if class_id_map:
        cmd += ["--class-id-map", class_id_map]

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


def step_stac(
    paths: TilePaths,
    cwd: Path,
    python_exe: str,
    variant: str,
    collection_id: str,
    overwrite: bool,
) -> None:
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
        "--variant",
        str(variant),
        "--collection-id",
        str(collection_id),
    ]
    _run(cmd, cwd=cwd)


def _to_repo_relative_url(path: Path, repo_root: Path) -> str:
    """Convert a filesystem path into a repo-relative URL-ish path.

    The viewer fetches assets via HTTP from the repo root (range_http_server.py).
    Absolute filesystem paths won't resolve in the browser, so we prefer paths
    relative to repo_root when possible.
    """

    try:
        rel = path.resolve().relative_to(repo_root.resolve())
        return rel.as_posix()
    except Exception:
        # If the output is outside the repo root, we cannot make it work in the
        # browser without changing the server root. Keep the original string.
        return path.as_posix()


def write_tiles_index(
    index_path: Path,
    tile_ids: Iterable[str],
    output_root: Path,
    output_prefix: str,
    repo_root: Path,
) -> None:
    tiles = []
    for tile_id in sorted(tile_ids, key=lambda x: int(x) if x.isdigit() else x):
        out_dir = output_root / f"{output_prefix}{tile_id}"
        base_url = _to_repo_relative_url(out_dir, repo_root=repo_root)
        thumb_url = _to_repo_relative_url(out_dir / f"thumbnail_531_tile_{tile_id}.png", repo_root=repo_root)
        tiles.append(
            {
                "tile_id": str(tile_id),
                "base_path": base_url + "/",
                "thumbnail": thumb_url,
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

    variant = str(args.variant).lower()
    stac_collection_id = str(args.stac_collection_id).strip() or f"wombat-fvc-{variant}"

    resolved_valid_classes = str(args.valid_classes).strip()
    if not resolved_valid_classes:
        if variant in {"low", "low_sparse"}:
            resolved_valid_classes = "0,1,2"
        elif variant in {"medium", "medium_sparse"}:
            resolved_valid_classes = "0,1,2,3"
        elif variant in {"dense"}:
            resolved_valid_classes = "0,1,2,3,4"
        else:
            resolved_valid_classes = "0,1,2,3"

    for tile_id in sorted(tile_ids, key=lambda x: int(x) if x.isdigit() else x):
        src = tiles[tile_id]
        paths = build_tile_paths(tile_id, output_root=output_root, output_prefix=args.output_prefix)

        try:
            paths.out_dir.mkdir(parents=True, exist_ok=True)

            stage_mode = str(args.stage_mode)
            _stage_input(src, paths.predictor_raw, mode=stage_mode, overwrite=args.overwrite)

            input_raster = src if stage_mode == "none" else paths.predictor_raw

            # Reproject/COG are always written into the output directory (for the viewer/STAC).
            if not ((paths.predictor_epsg4326.exists() or paths.predictor_cog.exists()) and not args.overwrite):
                _ensure_parent(paths.predictor_epsg4326)
                _run(
                    [
                        "gdalwarp",
                        "-overwrite" if args.overwrite else "-overwrite",
                        "-t_srs",
                        "EPSG:4326",
                        "-r",
                        "bilinear",
                        "-multi",
                        "-wo",
                        "NUM_THREADS=ALL_CPUS",
                        str(input_raster),
                        str(paths.predictor_epsg4326),
                    ],
                    cwd=cwd,
                )

            step_build_cog(paths, cwd=cwd, overwrite=args.overwrite)

            # Optionally delete the heavy epsg4326 intermediate once COG is built.
            if args.cleanup_intermediates and paths.predictor_cog.exists():
                try:
                    if paths.predictor_epsg4326.exists():
                        paths.predictor_epsg4326.unlink()
                except Exception:
                    pass

            step_inference(
                paths,
                cwd=cwd,
                python_exe=python_exe,
                model_path=model_path,
                input_raster=input_raster,
                in_channels=int(args.in_channels),
                valid_classes=resolved_valid_classes,
                variant=variant,
                class_id_map=str(args.class_id_map),
                write_mask_tif=bool(args.write_mask_tif),
                write_shp=bool(args.write_shp),
                overwrite=args.overwrite,
            )
            step_thumbnail(paths, cwd=cwd, python_exe=python_exe, overwrite=args.overwrite)

            # If we now write a tile-specific thumbnail name, remove the legacy one
            # so each output folder stays tidy.
            if args.cleanup_intermediates:
                try:
                    legacy_thumb = paths.out_dir / "thumbnail_531.png"
                    if legacy_thumb.exists():
                        legacy_thumb.unlink()
                except Exception:
                    pass
            step_stac(
                paths,
                cwd=cwd,
                python_exe=python_exe,
                variant=variant,
                collection_id=stac_collection_id,
                overwrite=args.overwrite,
            )

            # Optionally delete staged raw tile to save disk.
            if args.cleanup_intermediates:
                try:
                    # If a previous run staged the raw tile, remove it so the
                    # output folder contains only viewer-friendly products.
                    if paths.predictor_raw.exists() or paths.predictor_raw.is_symlink():
                        paths.predictor_raw.unlink()
                except Exception:
                    pass

            processed.append(tile_id)
            print(f"OK tile {tile_id}: {paths.out_dir}")

        except Exception as exc:
            print(f"FAIL tile {tile_id}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                raise

    if args.write_tiles_index:
        index_path = Path(args.write_tiles_index)
        write_tiles_index(
            index_path,
            processed,
            output_root=output_root,
            output_prefix=args.output_prefix,
            repo_root=cwd,
        )
        print(f"Wrote tiles index: {index_path}")

        # Also write a dataset-local tiles_index.json inside the output root.
        # This avoids collisions when running multiple datasets sequentially
        # (repo-root tiles_index.json would otherwise be overwritten each time).
        try:
            output_root_index = (output_root / "tiles_index.json")
            if output_root_index.resolve() != index_path.resolve():
                write_tiles_index(
                    output_root_index,
                    processed,
                    output_root=output_root,
                    output_prefix=args.output_prefix,
                    repo_root=cwd,
                )
                print(f"Wrote tiles index: {output_root_index}")
        except Exception:
            pass

    print(f"Done. Processed {len(processed)} tiles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
