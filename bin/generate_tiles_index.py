#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIEWER_ROOT = REPO_ROOT / "phase_3_models" / "unet_site_models" / "wombat_mappingAI_viewer"
TILE_DIR_PATTERN = re.compile(r"tile(?P<tile_id>\d+)$")
SOURCE_TILE_PATTERN = re.compile(r"tiles_multispectral[._](?P<tile_id>\d+)\.tif$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a dataset-local tiles_index.json for the composition viewer."
    )
    parser.add_argument(
        "dataset",
        help="Dataset name under the default viewer root, or a path to a dataset directory.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="Repository root used to compute browser-relative paths.",
    )
    parser.add_argument(
        "--viewer-root",
        default=str(DEFAULT_VIEWER_ROOT),
        help="Viewer root containing dataset directories.",
    )
    parser.add_argument(
        "--output",
        help="Optional output path. Defaults to <dataset>/tiles_index.json.",
    )
    parser.add_argument(
        "--overview-output",
        help="Optional output path. Defaults to <dataset>/overview_footprints.json.",
    )
    parser.add_argument(
        "--source-tile-dir",
        help=(
            "Optional source tile directory used to filter the published tile index to only original input tiles. "
            "The path may point at the tile directory itself or a parent folder that contains it."
        ),
    )
    return parser.parse_args()


def resolve_dataset_dir(dataset_arg: str, viewer_root: Path) -> Path:
    candidate = Path(dataset_arg)
    if candidate.exists():
        return candidate.resolve()

    dataset_dir = (viewer_root / dataset_arg).resolve()
    if dataset_dir.exists():
        return dataset_dir

    raise SystemExit(f"Dataset directory not found: {dataset_arg}")


def to_repo_relative_url(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception as exc:
        raise SystemExit(f"Path is outside repo root and cannot be published by the viewer: {path}") from exc


def infer_tile_id(tile_dir: Path) -> str | None:
    match = TILE_DIR_PATTERN.search(tile_dir.name)
    if not match:
        return None
    return match.group("tile_id")


def collect_source_tile_ids(source_tile_dir: Path) -> set[str]:
    tile_ids: set[str] = set()

    for path in source_tile_dir.rglob("*"):
        if not path.is_file():
            continue
        match = SOURCE_TILE_PATTERN.search(path.name)
        if match:
            tile_ids.add(match.group("tile_id"))

    if not tile_ids:
        raise SystemExit(f"No source tile ids found under: {source_tile_dir}")

    return tile_ids


def collect_source_tile_paths(source_tile_dir: Path) -> dict[str, Path]:
    tile_paths: dict[str, Path] = {}

    for path in sorted(source_tile_dir.rglob("*")):
        if not path.is_file():
            continue
        match = SOURCE_TILE_PATTERN.search(path.name)
        if match:
            tile_paths[match.group("tile_id")] = path

    if not tile_paths:
        raise SystemExit(f"No source tile paths found under: {source_tile_dir}")

    return tile_paths


def find_thumbnail(tile_dir: Path, tile_id: str) -> Path | None:
    for name in (f"thumbnail_531_tile_{tile_id}.png", "thumbnail_531.png"):
        candidate = tile_dir / name
        if candidate.exists():
            return candidate
    return None


def find_stac_item(tile_dir: Path, tile_id: str) -> Path | None:
    candidate = tile_dir / "stac" / f"item_tile{tile_id}.json"
    if candidate.exists():
        return candidate
    return None


def find_predictor_cog(tile_dir: Path, tile_id: str) -> Path | None:
    candidate = tile_dir / f"predictor_tile_{tile_id}_epsg4326_cog.tif"
    if candidate.exists():
        return candidate
    return None


def load_bbox_from_stac(stac_item_path: Path) -> list[float] | None:
    try:
        payload = json.loads(stac_item_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    bbox = payload.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    try:
        return [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None


def load_bbox_from_predictor(predictor_path: Path) -> list[float] | None:
    try:
        with rasterio.open(predictor_path) as src:
            bounds = src.bounds
            crs = src.crs
    except Exception:
        return None

    if crs and str(crs).upper() != "EPSG:4326":
        try:
            left, bottom, right, top = transform_bounds(crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top, densify_pts=21)
            return [float(left), float(bottom), float(right), float(top)]
        except Exception:
            return None

    return [float(bounds.left), float(bounds.bottom), float(bounds.right), float(bounds.top)]


def has_predictions(tile_dir: Path, tile_id: str) -> bool:
    return any(
        (tile_dir / name).exists()
        for name in (f"predictions_tile_{tile_id}.geojson", "predictions.geojson")
    )


def find_predictions_geojson(tile_dir: Path, tile_id: str) -> Path | None:
    for name in (f"predictions_tile_{tile_id}.geojson", "predictions.geojson"):
        candidate = tile_dir / name
        if candidate.exists():
            return candidate
    return None


def iter_geojson_positions(value):
    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and all(isinstance(coord, (int, float)) for coord in value[:2]):
            yield float(value[0]), float(value[1])
            return
        for item in value:
            yield from iter_geojson_positions(item)


def load_bbox_from_predictions(predictions_path: Path) -> list[float] | None:
    try:
        payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    xs = []
    ys = []
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list):
        return None

    for feature in features:
        geometry = feature.get("geometry") if isinstance(feature, dict) else None
        if not isinstance(geometry, dict):
            continue
        for x, y in iter_geojson_positions(geometry.get("coordinates")):
            xs.append(x)
            ys.append(y)

    if not xs or not ys:
        return None

    return [min(xs), min(ys), max(xs), max(ys)]


def get_tile_asset_paths(tile_dir: Path, tile_id: str) -> list[Path]:
    asset_paths: list[Path] = []

    predictions_geojson = find_predictions_geojson(tile_dir, tile_id)
    if predictions_geojson is not None:
        asset_paths.append(predictions_geojson)

    predictor_cog = find_predictor_cog(tile_dir, tile_id)
    if predictor_cog is not None:
        asset_paths.append(predictor_cog)

    thumbnail = find_thumbnail(tile_dir, tile_id)
    if thumbnail is not None:
        asset_paths.append(thumbnail)

    stac_item = find_stac_item(tile_dir, tile_id)
    if stac_item is not None:
        asset_paths.append(stac_item)

    return asset_paths


def compute_asset_version(asset_paths: list[Path]) -> str:
    latest_mtime = 0
    for path in asset_paths:
        try:
            latest_mtime = max(latest_mtime, int(path.stat().st_mtime))
        except OSError:
            continue
    return str(latest_mtime or 0)


def build_tiles_payload(
    dataset_dir: Path,
    repo_root: Path,
    allowed_tile_ids: set[str] | None = None,
    source_tile_dir: Path | None = None,
    source_tile_paths: dict[str, Path] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    tiles = []
    footprints = []
    footprint_tile_ids: set[str] = set()
    skipped = []
    asset_paths: list[Path] = []

    for tile_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        tile_id = infer_tile_id(tile_dir)
        if tile_id is None:
            skipped.append(f"skip {tile_dir.name}: cannot infer tile id")
            continue
        if allowed_tile_ids is not None and tile_id not in allowed_tile_ids:
            skipped.append(f"skip {tile_dir.name}: tile id {tile_id} not present in source tile inventory")
            continue
        if not has_predictions(tile_dir, tile_id):
            skipped.append(f"skip {tile_dir.name}: missing prediction GeoJSON")
            continue

        entry = {
            "tile_id": tile_id,
            "base_path": to_repo_relative_url(tile_dir, repo_root) + "/",
        }

        thumbnail = find_thumbnail(tile_dir, tile_id)
        if thumbnail is not None:
            entry["thumbnail"] = to_repo_relative_url(thumbnail, repo_root)

        asset_paths.extend(get_tile_asset_paths(tile_dir, tile_id))
        stac_item = find_stac_item(tile_dir, tile_id)
        predictor_cog = find_predictor_cog(tile_dir, tile_id)
        predictions_geojson = find_predictions_geojson(tile_dir, tile_id)
        bbox = load_bbox_from_stac(stac_item) if stac_item is not None else None
        if bbox is None and predictor_cog is not None:
            bbox = load_bbox_from_predictor(predictor_cog)
        if bbox is None and predictions_geojson is not None:
            bbox = load_bbox_from_predictions(predictions_geojson)
        if bbox is not None:
            footprints.append(
                {
                    "tile_id": tile_id,
                    "bbox": bbox,
                }
            )
            footprint_tile_ids.add(tile_id)

        tiles.append(entry)

    if allowed_tile_ids is not None and source_tile_paths is not None:
        for tile_id in sorted(allowed_tile_ids, key=lambda value: int(value) if str(value).isdigit() else str(value)):
            if tile_id in footprint_tile_ids:
                continue

            source_tile_path = source_tile_paths.get(tile_id)
            if source_tile_path is None:
                skipped.append(f"skip source tile {tile_id}: source raster path not found")
                continue

            bbox = load_bbox_from_predictor(source_tile_path)
            if bbox is None:
                skipped.append(f"skip source tile {tile_id}: could not read source raster bounds")
                continue

            footprints.append(
                {
                    "tile_id": tile_id,
                    "bbox": bbox,
                }
            )
            footprint_tile_ids.add(tile_id)
            asset_paths.append(source_tile_path)

    if not tiles:
        raise SystemExit(f"No publishable tile directories found under: {dataset_dir}")

    asset_version = compute_asset_version(asset_paths)

    tiles_payload = {
        "schema_version": 1,
        "tiles": sorted(tiles, key=lambda item: int(item["tile_id"]) if str(item["tile_id"]).isdigit() else str(item["tile_id"])),
        "_meta": {
            "dataset": dataset_dir.name,
            "tile_count": len(tiles),
            "asset_version": asset_version,
            "skipped": skipped,
        },
    }

    if source_tile_dir is not None and allowed_tile_ids is not None:
        tiles_payload["_meta"]["source_tile_dir"] = str(source_tile_dir)
        tiles_payload["_meta"]["source_tile_count"] = len(allowed_tile_ids)

    overview_payload = {
        "schema_version": 1,
        "dataset": dataset_dir.name,
        "asset_version": asset_version,
        "footprints": sorted(
            footprints,
            key=lambda item: int(item["tile_id"]) if str(item["tile_id"]).isdigit() else str(item["tile_id"]),
        ),
    }

    return tiles_payload, overview_payload


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    viewer_root = Path(args.viewer_root).resolve()
    dataset_dir = resolve_dataset_dir(args.dataset, viewer_root)
    source_tile_dir = Path(args.source_tile_dir).resolve() if args.source_tile_dir else None
    source_tile_paths = collect_source_tile_paths(source_tile_dir) if source_tile_dir else None
    allowed_tile_ids = set(source_tile_paths) if source_tile_paths else None
    output_path = Path(args.output).resolve() if args.output else dataset_dir / "tiles_index.json"
    overview_output_path = Path(args.overview_output).resolve() if args.overview_output else dataset_dir / "overview_footprints.json"

    payload, overview_payload = build_tiles_payload(
        dataset_dir,
        repo_root,
        allowed_tile_ids=allowed_tile_ids,
        source_tile_dir=source_tile_dir,
        source_tile_paths=source_tile_paths,
    )
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    overview_output_path.write_text(json.dumps(overview_payload, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {output_path} with {payload['_meta']['tile_count']} tiles", file=sys.stderr)
    print(f"Wrote {overview_output_path} with {len(overview_payload['footprints'])} footprints", file=sys.stderr)
    for message in payload["_meta"]["skipped"]:
        print(message, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())