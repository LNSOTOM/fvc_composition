#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a GeoPackage vector layer to PMTiles using ogr2ogr."
    )
    parser.add_argument("--input-gpkg", required=True, help="Input GeoPackage path.")
    parser.add_argument("--output-pmtiles", required=True, help="Output PMTiles path.")
    parser.add_argument(
        "--layer",
        default=None,
        help="Optional input layer name. Defaults to all layers supported by ogr2ogr.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output PMTiles if it already exists.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional PMTiles name metadata.",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Optional PMTiles description metadata.",
    )
    parser.add_argument(
        "--minzoom",
        type=int,
        default=None,
        help="Optional PMTiles minimum zoom metadata.",
    )
    parser.add_argument(
        "--maxzoom",
        type=int,
        default=None,
        help="Optional PMTiles maximum zoom metadata.",
    )
    return parser.parse_args()


def require_ogr2ogr() -> str:
    binary = shutil.which("ogr2ogr")
    if binary:
        return binary
    raise SystemExit("ogr2ogr is required but was not found on PATH.")


def build_command(args: argparse.Namespace, ogr2ogr_bin: str) -> list[str]:
    input_gpkg = Path(args.input_gpkg).expanduser().resolve()
    output_pmtiles = Path(args.output_pmtiles).expanduser().resolve()

    if not input_gpkg.exists():
        raise SystemExit(f"Input GeoPackage not found: {input_gpkg}")

    if output_pmtiles.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_pmtiles}. Pass --overwrite to replace it.")

    output_pmtiles.parent.mkdir(parents=True, exist_ok=True)

    cmd = [ogr2ogr_bin]
    if args.overwrite:
        cmd.append("-overwrite")

    cmd.extend(["-f", "PMTiles", str(output_pmtiles), str(input_gpkg)])

    if args.layer:
        cmd.append(str(args.layer))

    if args.name:
        cmd.extend(["-dsco", f"NAME={args.name}"])
    if args.description:
        cmd.extend(["-dsco", f"DESCRIPTION={args.description}"])
    if args.minzoom is not None:
        cmd.extend(["-dsco", f"MINZOOM={args.minzoom}"])
    if args.maxzoom is not None:
        cmd.extend(["-dsco", f"MAXZOOM={args.maxzoom}"])

    return cmd


def main() -> int:
    args = parse_args()
    ogr2ogr_bin = require_ogr2ogr()
    cmd = build_command(args, ogr2ogr_bin)

    subprocess.run(cmd, check=True)

    output_pmtiles = Path(args.output_pmtiles).expanduser().resolve()
    print(f"Wrote PMTiles: {output_pmtiles}")
    print(f"Size bytes: {output_pmtiles.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())