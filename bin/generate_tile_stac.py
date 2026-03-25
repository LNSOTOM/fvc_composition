#!/usr/bin/env python3
"""Generate minimal STAC JSON files for a tile output folder.

Creates a per-tile STAC folder alongside outputs (predictor COG + predictions GeoJSON),
mirroring the structure already used for tile 22.

Example:
  python bin/generate_tile_stac.py \
    --tile-id 55 \
    --tile-dir phase_3_models/unet_site_models/wombat_predictions_stitch_medium_1024_120ep_raw_bestmodel_55_tile55
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import rasterio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate STAC metadata for a tile")
    parser.add_argument("--tile-id", required=True, help="Tile id string (e.g. 22, 55)")
    parser.add_argument("--tile-dir", required=True, help="Directory containing predictor/predictions outputs")
    parser.add_argument(
        "--variant",
        choices=["low", "low_sparse", "medium", "medium_sparse", "dense"],
        default="medium",
        help="Dataset variant label used for titles, ids, and default class lists (default: medium)",
    )
    parser.add_argument(
        "--collection-id",
        default="wombat-fvc-medium",
        help="STAC collection id (default: wombat-fvc-medium)",
    )
    parser.add_argument(
        "--catalog-id",
        default="wombat-fvc-catalog",
        help="STAC catalog id (default: wombat-fvc-catalog)",
    )
    parser.add_argument(
        "--license",
        default="proprietary",
        help="Collection license (default: proprietary)",
    )
    parser.add_argument(
        "--provider",
        default="LNSOTOM",
        help="Provider name (default: LNSOTOM)",
    )
    return parser.parse_args()


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _polygon_from_bbox(bbox: list[float]) -> dict:
    minx, miny, maxx, maxy = bbox
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [minx, maxy],
                [minx, miny],
                [maxx, miny],
                [maxx, maxy],
                [minx, maxy],
            ]
        ],
    }


def main() -> int:
    args = parse_args()

    tile_id = str(args.tile_id)
    tile_dir = Path(args.tile_dir)
    if not tile_dir.exists():
        raise FileNotFoundError(f"Tile dir not found: {tile_dir}")

    predictor_name = f"predictor_tile_{tile_id}_epsg4326_cog.tif"
    predictions_name_new = f"predictions_tile_{tile_id}.geojson"
    predictions_name_legacy = "predictions.geojson"
    mask_name_new = f"predictions_mask_tile_{tile_id}.tif"
    mask_name_legacy = "predictions_mask.tif"

    predictor_path = tile_dir / predictor_name
    predictions_path_new = tile_dir / predictions_name_new
    predictions_path_legacy = tile_dir / predictions_name_legacy
    predictions_path = predictions_path_new if predictions_path_new.exists() else predictions_path_legacy

    mask_path_new = tile_dir / mask_name_new
    mask_path_legacy = tile_dir / mask_name_legacy
    mask_path = mask_path_new if mask_path_new.exists() else mask_path_legacy

    predictions_name = predictions_path.name
    mask_name = mask_path.name

    if not predictor_path.exists():
        raise FileNotFoundError(f"Predictor COG not found: {predictor_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(
            "Predictions GeoJSON not found (tried both naming schemes): "
            f"{predictions_path_new} and {predictions_path_legacy}"
        )

    stac_dir = tile_dir / "stac"
    stac_dir.mkdir(parents=True, exist_ok=True)

    timestamp = _utc_now_iso()

    with rasterio.open(predictor_path) as src:
        bounds = src.bounds
        bbox = [
            round(float(bounds.left), 7),
            round(float(bounds.bottom), 7),
            round(float(bounds.right), 7),
            round(float(bounds.top), 7),
        ]

        transform = src.transform
        proj_transform = [
            float(transform.c),
            float(transform.a),
            float(transform.b),
            float(transform.f),
            float(transform.d),
            float(transform.e),
        ]

        epsg = None
        try:
            if src.crs is not None:
                epsg = src.crs.to_epsg()
        except Exception:
            epsg = None

        dtype = str(src.dtypes[0]) if src.dtypes else "float32"
        nodata = float(src.nodata) if src.nodata is not None else None

        eo_bands = [
            {"name": f"B{i}", "description": f"Multispectral band {i}"} for i in range(1, src.count + 1)
        ]
        raster_bands = [
            {"data_type": dtype, **({"nodata": nodata} if nodata is not None else {})}
            for _ in range(src.count)
        ]

        proj_shape = [int(src.height), int(src.width)]

    variant = str(args.variant).lower()
    if variant in {"low", "low_sparse"}:
        default_class_ids = [0, 1, 2]
        default_class_names = ["BE", "NPV", "PV"]
    else:
        # medium_sparse + medium + dense default to the 4-class scheme.
        default_class_ids = [0, 1, 2, 3]
        default_class_names = ["BE", "NPV", "PV", "SI"]

    item_id = f"wombat-{variant}-tile{tile_id}"
    item_filename = f"item_tile{tile_id}.json"

    catalog = {
        "stac_version": "1.1.0",
        "stac_extensions": [],
        "id": args.catalog_id,
        "type": "Catalog",
        "title": "Wombat FVC Catalog",
        "description": "STAC catalog for Wombat FVC outputs (predictor COG and prediction polygons).",
        "links": [
            {"rel": "root", "href": "catalog.json", "type": "application/json"},
            {"rel": "self", "href": "catalog.json", "type": "application/json"},
            {"rel": "child", "href": "collection.json", "type": "application/json"},
            {"rel": "data", "href": "collections.json", "type": "application/json"},
        ],
    }

    collection = {
        "stac_version": "1.1.0",
        "stac_extensions": [],
        "type": "Collection",
        "id": args.collection_id,
        "title": f"Wombat FVC Predictions ({variant.capitalize()} Tiles)",
        "description": f"Predictor COG and FVC prediction polygons for Wombat {variant} tile {tile_id}.",
        "keywords": [
            "FVC",
            "fractional vegetation cover",
            "UAS",
            "multispectral",
            "COG",
            "GeoJSON",
        ],
        "license": args.license,
        "providers": [
            {
                "name": args.provider,
                "roles": ["producer", "processor", "host"],
            }
        ],
        "extent": {
            "spatial": {"bbox": [bbox]},
            "temporal": {"interval": [[timestamp, timestamp]]},
        },
        "summaries": {
            "fvc:class_id": default_class_ids,
            "fvc:class_name": default_class_names,
            "data_type": [dtype],
            "platform": ["UAS"],
        },
        "links": [
            {"rel": "root", "href": "catalog.json", "type": "application/json"},
            {"rel": "self", "href": "collection.json", "type": "application/json"},
            {"rel": "parent", "href": "catalog.json", "type": "application/json"},
            {"rel": "item", "href": item_filename, "type": "application/geo+json"},
            {"rel": "items", "href": "items.json", "type": "application/geo+json"},
        ],
    }

    item = {
        "stac_version": "1.1.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/projection/v1.1.0/schema.json",
            "https://stac-extensions.github.io/raster/v1.1.0/schema.json",
            "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
        ],
        "type": "Feature",
        "id": item_id,
        "bbox": bbox,
        "geometry": _polygon_from_bbox(bbox),
        "properties": {
            "datetime": timestamp,
            "created": timestamp,
            "updated": timestamp,
            "title": f"Wombat {variant} tile {tile_id}",
            **({"proj:epsg": int(epsg)} if epsg is not None else {}),
            "proj:shape": proj_shape,
            "proj:transform": proj_transform,
            "fvc:classes": default_class_names,
        },
        "collection": args.collection_id,
        "assets": {
            "predictor": {
                "href": f"../{predictor_name}",
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "roles": ["data", "reflectance"],
                "title": "Predictor COG",
                "eo:bands": eo_bands,
                "raster:bands": raster_bands,
            },
            "predictions": {
                "href": f"../{predictions_name}",
                "type": "application/geo+json",
                "roles": ["data", "labels"],
                "title": "FVC prediction polygons",
            },
            **(
                {
                    "mask": {
                        "href": f"../{mask_name}",
                        "type": "image/tiff; application=geotiff",
                        "roles": ["data", "labels"],
                        "title": "Predicted class mask (GeoTIFF)",
                    }
                }
                if mask_path.exists()
                else {}
            ),
        },
        "links": [
            {"rel": "root", "href": "catalog.json", "type": "application/json"},
            {"rel": "self", "href": item_filename, "type": "application/geo+json"},
            {"rel": "parent", "href": "collection.json", "type": "application/json"},
            {"rel": "collection", "href": "collection.json", "type": "application/json"},
        ],
    }

    items = {
        "type": "FeatureCollection",
        "features": [
            {
                **item,
                # In items.json, the tile22 version omits detailed band metadata.
                # Keep the full object here for simplicity/consistency.
            }
        ],
        "links": [
            {"rel": "root", "href": "catalog.json", "type": "application/json"},
            {"rel": "self", "href": "items.json", "type": "application/geo+json"},
            {"rel": "parent", "href": "collection.json", "type": "application/json"},
        ],
    }

    collections = {
        "collections": [
            {
                "type": "Collection",
                "id": args.collection_id,
                "stac_version": "1.1.0",
                "description": collection["description"],
                "links": [
                    {"href": "catalog.json", "rel": "root", "type": "application/json"},
                    {"href": "catalog.json", "rel": "parent", "type": "application/json"},
                    {"href": "collection.json", "rel": "self", "type": "application/json"},
                    {"href": "items.json", "rel": "items", "type": "application/geo+json"},
                ],
                "title": collection["title"],
                "extent": collection["extent"],
                "license": collection["license"],
                "providers": collection["providers"],
            }
        ],
        "links": [
            {"href": "catalog.json", "rel": "root", "type": "application/json"},
            {"href": "collections.json", "rel": "self", "type": "application/json"},
        ],
    }

    def write_json(path: Path, data: dict) -> None:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    write_json(stac_dir / "catalog.json", catalog)
    write_json(stac_dir / "collection.json", collection)
    write_json(stac_dir / "collections.json", collections)
    write_json(stac_dir / item_filename, item)
    write_json(stac_dir / "items.json", items)

    print(f"Wrote STAC to: {stac_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
