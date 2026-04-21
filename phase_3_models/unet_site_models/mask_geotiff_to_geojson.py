#!/usr/bin/env python3
"""Convert a class-mask GeoTIFF to GeoJSON polygons."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

from phase_3_models.unet_site_models.inference_raster_to_geojson import (
    VARIANT_PRESETS,
    mask_to_polygons,
    parse_valid_classes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a class-mask GeoTIFF to GeoJSON polygons")
    parser.add_argument("--input-mask", required=True, help="Path to the single-band class-mask GeoTIFF")
    parser.add_argument("--output-geojson", required=True, help="Output GeoJSON path")
    parser.add_argument(
        "--output-crs",
        default="EPSG:4326",
        help="Target CRS for the GeoJSON output. Defaults to EPSG:4326 for GeoJSON/QGIS compatibility.",
    )
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_PRESETS.keys()),
        default="medium",
        help="Class naming preset for properties.class_name (default: medium)",
    )
    parser.add_argument(
        "--valid-classes",
        type=str,
        default=None,
        help="Comma-separated class ids to polygonize. Defaults to the classes from --variant.",
    )
    parser.add_argument(
        "--mask-nodata",
        type=int,
        default=None,
        help="Override the raster nodata value to exclude from polygonization.",
    )
    parser.add_argument("--include-class-zero", action="store_true", default=True)
    parser.add_argument("--exclude-class-zero", action="store_false", dest="include_class_zero")
    parser.add_argument("--min-polygon-area", type=float, default=0.0)
    parser.add_argument("--sieve-size", type=int, default=0)
    parser.add_argument("--si-min-polygon-area", type=float, default=0.0)
    parser.add_argument("--si-max-polygon-area", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_mask = Path(args.input_mask)
    output_geojson = Path(args.output_geojson)

    if not input_mask.exists():
        raise FileNotFoundError(f"Input mask not found: {input_mask}")

    output_geojson.parent.mkdir(parents=True, exist_ok=True)

    preset = VARIANT_PRESETS[args.variant]
    class_id_to_name = {int(v): str(k) for k, v in preset["class_labels"].items()}
    preset_valid = ",".join(str(v) for v in sorted(preset["class_labels"].values()))
    valid_classes = parse_valid_classes(args.valid_classes if args.valid_classes is not None else preset_valid)

    with rasterio.open(input_mask) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata_value = args.mask_nodata if args.mask_nodata is not None else src.nodata

    if nodata_value is not None:
        if isinstance(nodata_value, float) and np.isnan(nodata_value):
            mask = np.where(np.isnan(mask), np.nan, mask)
        else:
            valid_classes = [class_id for class_id in valid_classes if int(class_id) != int(nodata_value)]

    gdf = mask_to_polygons(
        mask=mask,
        transform=transform,
        crs=crs,
        include_class_zero=args.include_class_zero,
        min_polygon_area=float(args.min_polygon_area),
        valid_classes=valid_classes,
        sieve_size=int(args.sieve_size),
        si_min_polygon_area=float(args.si_min_polygon_area),
        si_max_polygon_area=float(args.si_max_polygon_area),
        class_id_to_name=class_id_to_name,
    )

    output_crs = str(args.output_crs).strip()
    if output_crs:
        gdf = gdf.to_crs(output_crs)

    gdf.to_file(output_geojson, driver="GeoJSON")

    print(f"Wrote GeoJSON: {output_geojson}")
    print(f"Input mask: {input_mask}")
    print(f"Polygon features: {len(gdf)}")
    print(f"Classes polygonized: {valid_classes}")
    if output_crs:
        print(f"Output CRS: {gdf.crs}")
    if nodata_value is not None:
        print(f"Excluded nodata value: {nodata_value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())